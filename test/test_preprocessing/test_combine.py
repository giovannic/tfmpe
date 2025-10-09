"""Tests for combine_tokens functionality.

Tests verify combining multiple Tokens objects for dataset accumulation.
"""

import jax.numpy as jnp
import pytest
from tfmpe.preprocessing import Tokens, combine_tokens

@pytest.fixture
def simple_pytree_sample2():
    """Simple pytree with sample_shape=(2,), same n_tokens."""
    return {
        'mu': jnp.array([[[1.0]], [[2.0]]]),  # (2, 1, 1)
        'theta': jnp.array([[[3.0]], [[4.0]]])  # (2, 1, 1)
    }


@pytest.fixture
def simple_pytree_sample3():
    """Simple pytree with sample_shape=(3,), same n_tokens."""
    return {
        'mu': jnp.array([[[5.0]], [[6.0]], [[7.0]]]),  # (3, 1, 1)
        'theta': jnp.array([[[8.0]], [[9.0]], [[10.0]]])  # (3, 1, 1)
    }


def test_combine_same_n_tokens(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test combining Tokens with same n_tokens (no padding)."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=['mu'],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=['mu'],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check sample dimension is concatenated: 2 + 3 = 5
    assert combined.data.shape[0] == 5

    # Check n_tokens dimension is preserved (both have 2 tokens)
    assert combined.data.shape[1] == tokens1.data.shape[1]
    assert combined.data.shape[1] == tokens2.data.shape[1]

    # Check batch dimension preserved
    assert combined.data.shape[2] == tokens1.data.shape[2]


def test_combine_different_n_tokens():
    """Test combining Tokens with different n_tokens (padding)."""
    pytree1 = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1) -> 2 tokens
        'b': jnp.array([[[3.0]]])  # (1, 1, 1) -> 1 token
    }
    pytree2 = {
        'a': jnp.array([[[4.0], [5.0], [6.0]]]),  # (1, 3, 1) -> 3 tokens
        'b': jnp.array([[[7.0], [8.0]]])  # (1, 2, 1) -> 2 tokens
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        condition=[],
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'a': 1, 'b': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check n_tokens padded to max
    max_n = max(tokens1.data.shape[1], tokens2.data.shape[1])
    assert combined.data.shape[1] == max_n

    # Check sample dimension concatenated: 1 + 1 = 2
    assert combined.data.shape[0] == 2


def test_combine_labels_same_n_tokens(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test labels when n_tokens is same."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Labels should be identical since same structure
    # First samples (2) should match tokens1
    assert jnp.array_equal(
        combined.labels[0:2],
        tokens1.labels
    )

    # Next samples (3) should match tokens2
    assert jnp.array_equal(
        combined.labels[2:5],
        tokens2.labels
    )

def test_combine_functional_inputs_both_present(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test functional_inputs combined when both present."""
    func_inputs1 = {
        'mu': jnp.array([[[0.0]], [[0.0]]]),
        'theta': jnp.array([[[1.0]], [[1.0]]])
    }

    func_inputs2 = {
        'mu': jnp.array([[[0.0]], [[0.0]], [[0.0]]]),
        'theta': jnp.array([[[2.0]], [[2.0]], [[2.0]]])
    }

    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=[],
        functional_inputs=func_inputs2,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check functional_inputs is not None
    assert combined.functional_inputs is not None

    # Check shape matches combined data
    assert combined.functional_inputs.shape == combined.data.shape


def test_combine_functional_inputs_error_one_absent(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test error when only one Tokens has functional_inputs."""
    func_inputs1 = {
        'mu': jnp.array([[[0.0]], [[0.0]]]),
        'theta': jnp.array([[[1.0]], [[1.0]]])
    }

    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    # Should raise error for mismatched functional_inputs
    with pytest.raises(ValueError, match="functional_inputs"):
        combine_tokens(tokens1, tokens2)


def test_combine_functional_inputs_both_absent(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test functional_inputs is None when both absent."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    assert combined.functional_inputs is None


def test_combine_functional_inputs_with_padding():
    """Test functional_inputs padded with FUNCTIONAL_INPUT_PAD_VALUE for smaller tokens."""
    pytree1 = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1)
    }
    pytree2 = {
        'a': jnp.array([[[3.0], [4.0], [5.0]]]),  # (1, 3, 1)
    }
    func_inputs1 = {
        'a': jnp.array([[[10.0], [20.0]]]),  # (1, 2, 1)
    }
    func_inputs2 = {
        'a': jnp.array([[[30.0], [40.0], [50.0]]]),  # (1, 3, 1)
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        condition=[],
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'a': 1},
        pad_to_even=False
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        condition=[],
        functional_inputs=func_inputs2,
        sample_ndims=1,
        batch_ndims={'a': 1},
        pad_to_even=False
    )

    combined = combine_tokens(tokens1, tokens2, pad_to_even=False)

    # Check functional_inputs is not None
    assert combined.functional_inputs is not None

    # Sample 0 (from tokens1): first 2 match, last is padded with 0.0
    assert jnp.allclose(
        combined.functional_inputs[0],
        jnp.array([[10.0], [20.0], [0.0]])
    )

    # Sample 1 (from tokens2): all positions match
    assert jnp.allclose(
        combined.functional_inputs[1],
        jnp.array([[30.0], [40.0], [50.0]])
    )


def test_combine_multiple_via_repeated_calls(
    simple_pytree_sample2,
):
    """Test combining >2 Tokens via repeated calls."""
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens3 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    # Combine three tokens
    combined12 = combine_tokens(tokens1, tokens2)
    combined123 = combine_tokens(combined12, tokens3)

    # Check sample dimension: 2 + 2 + 2 = 6
    assert combined123.data.shape[0] == 6


def test_combine_data_values_preserved(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test that actual data values are correctly concatenated."""
    tokens1, decoder = Tokens.from_pytree_with_decoder(
        simple_pytree_sample2,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1},
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=[],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1},
    )

    combined = combine_tokens(tokens1, tokens2)

    # Decode and check values
    combined_decoded = decoder(combined)

    # First samples should match tokens1
    assert jnp.array_equal(
        combined_decoded['mu'][0:2],
        simple_pytree_sample2['mu']
    )
    assert jnp.array_equal(
        combined_decoded['theta'][0:2],
        simple_pytree_sample2['theta']
    )

    # Next samples should match tokens2
    assert jnp.array_equal(
        combined_decoded['mu'][2:5],
        simple_pytree_sample3['mu']
    )
    assert jnp.array_equal(
        combined_decoded['theta'][2:5],
        simple_pytree_sample3['theta']
    )


def test_combine_padding_mask_set():
    """Test padding_mask is correctly set on combined Tokens."""
    pytree1 = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1) -> 2 tokens
        'b': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1) -> 2 tokens
    }
    pytree2 = {
        'a': jnp.array([[[3.0], [4.0], [5.0]]]),  # (1, 3, 1) -> 3 tokens
        'b': jnp.array([[[3.0], [4.0], [5.0]]]),  # (1, 3, 1) -> 3 tokens
    }

    tokens1 = Tokens.from_pytree(
        pytree1,
        condition=['a'],
        sample_ndims=1,
    )

    tokens2 = Tokens.from_pytree(
        pytree2,
        condition=['a'],
        sample_ndims=1,
    )

    combined = combine_tokens(tokens1, tokens2)

    # Padding mask should exist
    assert combined.padding_mask is not None

    # Shape should match sample and token dimensions
    assert combined.padding_mask.shape == (2, 6)

    # Sample 0 (from tokens1): first 2 tokens are 1, last is 0 for condition and target
    assert jnp.array_equal(combined.padding_mask[0], jnp.array([1, 1, 0, 1, 1, 0]))

    # Sample 1 (from tokens2): all tokens are 1
    assert jnp.array_equal(combined.padding_mask[1], jnp.array([1, 1, 1, 1, 1, 1]))


def test_combine_error_different_final_dimension_with_functional_inputs():
    """Test error when functional_inputs have different final dimensions.

    The final dimension of functional_inputs must be consistent since it
    represents the batch dimension. Different dimensions indicate
    inconsistent functional input specifications.
    """
    pytree = {
        'a': jnp.array([[[1.0], [2.0]]]),  # (1, 2, 1)
    }

    # functional_inputs with final dimension of size 1
    func_inputs1 = {
        'a': jnp.array([[[10.0], [20.0]]]),  # (1, 2, 1)
    }

    # functional_inputs with final dimension of size 2
    func_inputs2 = {
        'a': jnp.array([[[30.0, 31.0], [40.0, 41.0]]]),  # (1, 2, 2)
    }

    tokens1 = Tokens.from_pytree(
        pytree,
        condition=[],
        functional_inputs=func_inputs1,
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    tokens2 = Tokens.from_pytree(
        pytree,
        condition=[],
        functional_inputs=func_inputs2,
        sample_ndims=1,
        batch_ndims={'a': 1}
    )

    # Should raise error for mismatched final dimensions in functional_inputs
    with pytest.raises(ValueError, match="functional_inputs"):
        combine_tokens(tokens1, tokens2)

def test_combine_position_and_condition_fields(
    simple_pytree_sample2,
    simple_pytree_sample3,
):
    """Test position and condition fields are correctly combined."""
    # Create tokens with conditioning variables
    tokens1 = Tokens.from_pytree(
        simple_pytree_sample2,
        condition=['theta'],  # theta is a conditioning variable
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    tokens2 = Tokens.from_pytree(
        simple_pytree_sample3,
        condition=['theta'],
        sample_ndims=1,
        batch_ndims={'mu': 1, 'theta': 1}
    )

    combined = combine_tokens(tokens1, tokens2)

    # Check position field exists and has correct shape
    assert combined.position is not None
    assert combined.position.shape == combined.data.shape[:2]  # (sample, tokens)

    # Check condition field exists and has correct shape
    assert combined.condition is not None
    assert combined.condition.shape == combined.data.shape[:2]  # (sample, tokens)

    # Position should be concatenated along sample dimension
    # First samples (2) should match tokens1
    assert jnp.array_equal(
        combined.position[0:2],
        tokens1.position
    )

    # Next samples (3) should match tokens2
    assert jnp.array_equal(
        combined.position[2:5],
        tokens2.position
    )

    # Condition should be concatenated along sample dimension
    assert jnp.array_equal(
        combined.condition[0:2],
        tokens1.condition
    )

    assert jnp.array_equal(
        combined.condition[2:5],
        tokens2.condition
    )
