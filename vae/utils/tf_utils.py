from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn
from tensorflow.python.util import nest
import collections
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import constant_op

_transpose_batch_time = rnn._transpose_batch_time


class LenControlWrapperState(
    collections.namedtuple("LenControlWrapperState",
                           ("cell_state", "time"))):
    def clone(self, **kwargs):
        return super(LenControlWrapperState, self)._replace(**kwargs)


class LenControlWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with LenEmb.
    """

    def __init__(self,
                 cell,
                 seq_len,
                 len_embeddings,
                 cell_input_fn=None,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.

        Args:
          cell: An instance of `RNNCell`.
          alignment_inputs: inputs
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, alignment_input: array_ops.concat([inputs, alignment_input], -1)`.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.

        Raises:
          TypeError: `attention_layer_size` is not None and (`attention_mechanism`
            is a list but `attention_layer_size` is not; or vice versa).
          ValueError: if `attention_layer_size` is not None, `attention_mechanism`
            is a list, and its length does not match that of `attention_layer_size`.
        """
        super(LenControlWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError(
                "cell must be an RNNCell, saw type: %s" % type(cell).__name__)

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, len_embedding: array_ops.concat([inputs, len_embedding], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s"
                    % type(cell_input_fn).__name__)

        self._cell = cell
        self._seq_len = seq_len
        self._len_embeddings = len_embeddings
        self._cell_input_fn = cell_input_fn
        with ops.name_scope(name, "LenControlWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or array_ops.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing LenControlWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with ops.control_dependencies(self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                        lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size, tf.shape(self._seq_len)[0])]

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return LenControlWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]))

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of AlignmentWrapperState %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with ops.control_dependencies(self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return LenControlWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([batch_size], dtype=dtypes.int32))

    def call(self, inputs, state):
        if not isinstance(state, LenControlWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        time_left = tf.maximum(self._seq_len - 1 - state.time, 0)
        len_embedding_inputs = tf.nn.embedding_lookup(self._len_embeddings, time_left)
        cell_inputs = self._cell_input_fn(inputs, len_embedding_inputs)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        next_state = LenControlWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state)
        return cell_output, next_state


class AlignmentWrapperState(
    collections.namedtuple("AlignmentWrapperState",
                           ("cell_state", "time"))):
    def clone(self, **kwargs):
        return super(AlignmentWrapperState, self)._replace(**kwargs)


class AlignmentWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 alignment_input,
                 cell_input_fn=None,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.

        Args:
          cell: An instance of `RNNCell`.
          alignment_inputs: inputs
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, alignment_input: array_ops.concat([inputs, alignment_input], -1)`.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.

        Raises:
          TypeError: `attention_layer_size` is not None and (`attention_mechanism`
            is a list but `attention_layer_size` is not; or vice versa).
          ValueError: if `attention_layer_size` is not None, `attention_mechanism`
            is a list, and its length does not match that of `attention_layer_size`.
        """
        super(AlignmentWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError(
                "cell must be an RNNCell, saw type: %s" % type(cell).__name__)

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, alignment_input: array_ops.concat([inputs, alignment_input], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s"
                    % type(cell_input_fn).__name__)

        self._cell = cell
        self._alignment_input = alignment_input
        self._cell_input_fn = cell_input_fn
        with ops.name_scope(name, "AlignmentWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or array_ops.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing AlignmentWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with ops.control_dependencies(self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                        lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size, tf.shape(self._alignment_input)[0])]

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return AlignmentWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]))

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of AlignmentWrapperState %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with ops.control_dependencies(self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return AlignmentWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32))

    def call(self, inputs, state):
        if not isinstance(state, AlignmentWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        cell_inputs = self._cell_input_fn(inputs, self._alignment_input)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        next_state = AlignmentWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state)

        return cell_output, next_state
