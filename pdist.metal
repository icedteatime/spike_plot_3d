

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

kernel void pairwise_distances(device float* out,
                               device float2* in,
                               constant uint& length,
                               uint index [[thread_position_in_grid]]) {

    uint2 I = source_indices(length, index);

    out[index] = metal::distance(in[I.x], in[I.y]);
}
