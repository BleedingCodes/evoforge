import numpy as np

from evoforge.genome import Genome, INPUTS, OUTPUTS


def test_genome_forward_shape():
    rng = np.random.default_rng(1)
    genome = Genome.random(rng)
    output = genome.forward(np.zeros(INPUTS, dtype=np.float32))
    assert output.shape == (OUTPUTS,)


def test_mutation_preserves_shapes():
    rng = np.random.default_rng(2)
    genome = Genome.random(rng)
    child = genome.mutate(rng, rate=1.0, scale=0.2)
    assert child.w1.shape == genome.w1.shape
    assert child.w2.shape == genome.w2.shape
