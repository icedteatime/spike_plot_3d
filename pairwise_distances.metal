
/*
We have an original list of points:
    S = [p0, p1, p2, p3]

Indices into this list are source indices.

We also have a list of distances:
    // P = [d(S[0], S[1]), d(S[0], S[2]), d(S[0], S[3]), d(S[2], S[3]), d(S[1], S[2]), d(S[1], S[3])]
    P = [d(S[0], S[1]), d(S[0], S[2]), d(S[0], S[3]), d(S[1], S[2]), d(S[1], S[3]), d(S[2], S[3])]

Indices into this list are pair indices.
*/

uint2 pair_index_to_source_indices0(uint num_points, uint pair_index) {
    /*
    Given an index into a list of pairs, return the source indices.
    Since P[2] = d(S[0], S[3]), pair_index_to_source_indices(2) = (0, 3).

    First get the usual i,j coordinate.
    If it is above the diagonal, keep it.
    If it is on the diagonal or below, rotate it to the
    opposite corner so that it is above the diagonal.
    This means the list of pairs isn't based on the standard order.

    pair_index will range from 0 to num_points choose 2 - 1.
    */

    uint L = num_points - 1; // We are ignoring the diagonal
    uint i = pair_index / L;
    uint j = pair_index % L + 1;

    if (j < i + 1) {
        return uint2(num_points - i - 1, num_points - j);
    } else {
        return uint2(i, j);
    }
}

uint source_indices_to_pair_index0(uint num_points, uint num_pairs, uint2 source_indices) {
    /*
    Example: source_indices_to_pair_index((2, 3)) = 3
    */

    uint i = source_indices.x;
    uint j = source_indices.y;

    uint L = num_points - 1;

    // Reverse / and % operators.
    uint I = i * L + j - 1;

    if (I > num_pairs - 1) {
        I = (L - i + 1) * L - j;
    }

    return I;
}

uint2 pair_index_to_source_indices(uint num_points, uint num_pairs, uint pair_index) {
    /*
    Given an index into a list of pairs, return the source indices.
    Since P[2] = d(S[0], S[3]), pair_index_to_source_indices(2) = (0, 3).

    This implementation is based on solving for n in the n choose 2 formula.
        n * (n - 1) / 2
    */

    // change to "depth first" ordering of pairs
    pair_index = num_pairs - pair_index - 1;
    int i = metal::floor(0.5 + metal::sqrt(0.25 + 2*pair_index));
    int j = pair_index - i * (i - 1) / 2;

    return uint2(num_points - 1 - i, num_points - 1 - j);
}


uint source_indices_to_pair_index(uint num_points, uint num_pairs, uint2 source_indices) {
    /*
    Example: source_indices_to_pair_index((2, 3)) = 5

    Reverse of pair_index_to_source_indices.
    */

    uint i = num_points - 1 - source_indices.x;
    uint pi = i * (i - 1) / 2 + num_points - source_indices.y - 1;

    return num_pairs - pi - 1;
}

kernel void debug1(device uint2* out,
                   constant uint& num_points,
                   constant uint& num_pairs,
                   uint index [[thread_position_in_grid]]) {

    uint2 S = pair_index_to_source_indices(num_points, num_pairs, index);

    out[index] = source_indices_to_pair_index(num_points, num_pairs, S);
}

uint2 pair_for_source_index(uint source_index,
                            uint i) {
    /*
    Given a source index, get the source indices for its ith pairing.
    pair_for_source_index(2, 0) = (0, 2)
    pair_for_source_index(2, 1) = (2, 3)
    pair_for_source_index(2, 2) = (1, 2)

    Based on moving in an L shape over the upper triangle.

     0123
    0..x.
    1..x.
    2...x
    3....
    */

    return uint2((i < source_index)*i + (i >= source_index)*source_index,
                 source_index + (i >= source_index)*(i - source_index + 1));
}

kernel void pairwise_distances_forward(device float* out,
                                       device float2* points,
                                       constant uint& num_points,
                                       constant uint& num_pairs,
                                       uint index [[thread_position_in_grid]]) {

    uint2 S = pair_index_to_source_indices(num_points, num_pairs, index);

    out[index] = metal::distance(points[S.x], points[S.y] + 0.000001);
}

kernel void pairwise_distances_backward(device float2* out,
                                        device float2* points,
                                        constant uint& num_points,
                                        constant uint& num_pairs,
                                        device float* grad_div_distances,
                                        uint index [[thread_position_in_grid]]) {
    /*
    For each point, add up the gradient from all of its pairings.
    */

    // index goes from 0 to out.numel()
    if (index >= num_points) {
        return ;
    }

    float2 gradient = 0;
    for (uint i = 0; i < num_points - 1; i++) {
        uint2 S = pair_for_source_index(index, i);
        uint I = source_indices_to_pair_index(num_points, num_pairs, S);

        int polarity = 1;
        if (S.x < index) {
            polarity = -1;
        }

        // Derivative of Euclidean distance d(p1, p2) is (p1 - p2) / d(p1, p2)
        gradient += polarity * (points[S.x] - points[S.y]) * grad_div_distances[I];
    }

    out[index] = gradient;
}
