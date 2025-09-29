

uint2 source_indices(constant uint& length, uint pair_index) {
    /*
    Given an index into a list of pairs,
    return the indices of each item in the pair.

    Based on traversing a length x length matrix which
    represents all ordered pairs, but taking only the upper
    triangle which represents unordered pairs.

    First get the usual i,j coordinate.
    If it is above the diagonal, keep it.
    If it is on the diagonal or below, rotate it to the
    opposite corner so that it is above the diagonal.
    This means the list of pairs isn't based on the standard order.

    pair_index will range from 0 to length choose 2 - 1.
    */

    uint L = length - 1; // We are ignoring the diagonal
    uint i = pair_index / L;
    uint j = pair_index % L + 1;

    if (j < i + 1) {
        return uint2(length - i - 1, length - j);
    } else {
        return uint2(i, j);
    }
}

uint pairs_index(uint length, uint pair_count, uint2 source_indices1) {
    uint i = source_indices1.x;
    uint j = source_indices1.y;

    uint L = length - 1;

    // Reverse / and % operators.
    uint I = i * L + j - 1;

    if (I > pair_count - 1) {
        I = (L - i + 1) * L - j;
    }

    return I;
}

uint2 pair_for_index(uint index,
                     uint i) {
    return uint2((i < index)*i + (i >= index)*index,
                 index + (i >= index)*(i - index + 1));
}

kernel void source_indices_(device long2* out,
                            constant uint& length,
                            uint index [[thread_position_in_grid]]) {
    out[index] = (long2)source_indices(length, index);
}
kernel void pairs_index_(device uint* out,
                         constant uint& length,
                         constant uint& pair_count,
                         device uint2& source_indices1) {
    out[0] = pairs_index(length, pair_count, source_indices1);
}

kernel void pairwise_distances(device float* out,
                               device float2* in,
                               constant uint& length,
                               uint index [[thread_position_in_grid]]) {

    uint2 I = source_indices(length, index);

    out[index] = metal::distance(in[I.x], in[I.y]);
}

kernel void pairwise_distances_backward(device float2* out,
                                        constant uint& length,
                                        constant uint& pair_count,
                                        device float2* in,
                                        device float* distances,
                                        device float* grad_output,
                                        uint index [[thread_position_in_grid]]) {

    float2 gradient = 0;
    for (uint i = 0; i < length - 1; i++) {
        uint2 S = pair_for_index(index, i);
        uint I = pairs_index(length, pair_count, S);

        float polarity = 1;
        if (S.x < index) {
            polarity = -1;
        }

        gradient += (polarity * (in[S.x] - in[S.y]) / distances[I]) * grad_output[I];
    }

    out[index] = gradient;
}
