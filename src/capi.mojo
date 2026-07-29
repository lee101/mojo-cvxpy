"""Sparse canonicalization kernels exposed through a stable C ABI."""

from std.algorithm import parallelize
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime PARALLEL_ROWS = 65_536
comptime PARALLEL_NONZEROS = 1_000_000
comptime MAX_WORKERS = 8


def csr_matvec_rows_i64(
    data: FPtr,
    indices: IPtr,
    indptr: IPtr,
    vector: FPtr,
    result: FPtr,
    start: Int,
    stop: Int,
):
    comptime W = simdwidthof[DType.float64]()
    for row in range(start, stop):
        var cursor = Int(indptr[row])
        var end = Int(indptr[row + 1])
        var vector_end = end - (end - cursor) % W
        var totals = SIMD[DType.float64, W](0.0)
        while cursor < vector_end:
            var columns = indices.load[width=W, alignment=1](cursor)
            totals += (
                data.load[width=W, alignment=1](cursor)
                * vector.gather(columns)
            )
            cursor += W
        var total = totals.reduce_add()
        while cursor < end:
            total += data[cursor] * vector[Int(indices[cursor])]
            cursor += 1
        result[row] = total


def csr_matvec_rows_i32(
    data: FPtr,
    indices: I32Ptr,
    indptr: I32Ptr,
    vector: FPtr,
    result: FPtr,
    start: Int,
    stop: Int,
):
    comptime W = simdwidthof[DType.float64]()
    for row in range(start, stop):
        var cursor = Int(indptr[row])
        var end = Int(indptr[row + 1])
        var vector_end = end - (end - cursor) % W
        var totals = SIMD[DType.float64, W](0.0)
        while cursor < vector_end:
            var columns = indices.load[width=W, alignment=1](cursor)
            totals += (
                data.load[width=W, alignment=1](cursor)
                * vector.gather(columns)
            )
            cursor += W
        var total = totals.reduce_add()
        while cursor < end:
            total += data[cursor] * vector[Int(indices[cursor])]
            cursor += 1
        result[row] = total


@export("mcvx_csr_matvec")
def mcvx_csr_matvec(
    data_addr: Int,
    indices_addr: Int,
    indptr_addr: Int,
    vector_addr: Int,
    result_addr: Int,
    rows: Int,
    nonzeros: Int,
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var indptr = IPtr(unsafe_from_address=indptr_addr)
    var vector = FPtr(unsafe_from_address=vector_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var workers = (
        min(min(num_physical_cores(), MAX_WORKERS), rows)
        if rows >= PARALLEL_ROWS and nonzeros >= PARALLEL_NONZEROS
        else 1
    )

    @parameter
    def process(worker: Int):
        var start = worker * rows // workers
        var stop = (worker + 1) * rows // workers
        csr_matvec_rows_i64(
            data, indices, indptr, vector, result, start, stop
        )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mcvx_csr_matvec_i32")
def mcvx_csr_matvec_i32(
    data_addr: Int,
    indices_addr: Int,
    indptr_addr: Int,
    vector_addr: Int,
    result_addr: Int,
    rows: Int,
    nonzeros: Int,
) abi("C"):
    var data = FPtr(unsafe_from_address=data_addr)
    var indices = I32Ptr(unsafe_from_address=indices_addr)
    var indptr = I32Ptr(unsafe_from_address=indptr_addr)
    var vector = FPtr(unsafe_from_address=vector_addr)
    var result = FPtr(unsafe_from_address=result_addr)
    var workers = (
        min(min(num_physical_cores(), MAX_WORKERS), rows)
        if rows >= PARALLEL_ROWS and nonzeros >= PARALLEL_NONZEROS
        else 1
    )

    @parameter
    def process(worker: Int):
        var start = worker * rows // workers
        var stop = (worker + 1) * rows // workers
        csr_matvec_rows_i32(
            data, indices, indptr, vector, result, start, stop
        )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mcvx_csr_nonempty_rows")
def mcvx_csr_nonempty_rows(
    indptr_addr: Int,
    rows: Int,
    old_rows_addr: Int,
    reduced_indptr_addr: Int,
) abi("C") -> Int:
    var indptr = IPtr(unsafe_from_address=indptr_addr)
    var old_rows = IPtr(unsafe_from_address=old_rows_addr)
    var reduced_indptr = IPtr(unsafe_from_address=reduced_indptr_addr)
    var count = 0
    reduced_indptr[0] = 0
    for row in range(rows):
        var end = indptr[row + 1]
        if end > indptr[row]:
            old_rows[count] = Int64(row)
            count += 1
            reduced_indptr[count] = end
    return count


@export("mcvx_csr_nonempty_rows_i32")
def mcvx_csr_nonempty_rows_i32(
    indptr_addr: Int,
    rows: Int,
    old_rows_addr: Int,
    reduced_indptr_addr: Int,
) abi("C") -> Int:
    var indptr = I32Ptr(unsafe_from_address=indptr_addr)
    var old_rows = IPtr(unsafe_from_address=old_rows_addr)
    var reduced_indptr = I32Ptr(unsafe_from_address=reduced_indptr_addr)
    var count = 0
    reduced_indptr[0] = 0
    for row in range(rows):
        var end = indptr[row + 1]
        if end > indptr[row]:
            old_rows[count] = Int64(row)
            count += 1
            reduced_indptr[count] = end
    return count


@export("mcvx_csc_compact")
def mcvx_csc_compact(
    data_addr: Int,
    row_indices_addr: Int,
    col_indptr_addr: Int,
    rows: Int,
    cols: Int,
    nonzeros: Int,
    scratch_addr: Int,
    reduced_data_addr: Int,
    reduced_indices_addr: Int,
    reduced_indptr_addr: Int,
    old_rows_addr: Int,
) abi("C") -> Int:
    var data = FPtr(unsafe_from_address=data_addr)
    var row_indices = IPtr(unsafe_from_address=row_indices_addr)
    var col_indptr = IPtr(unsafe_from_address=col_indptr_addr)
    var scratch = IPtr(unsafe_from_address=scratch_addr)
    var reduced_data = FPtr(unsafe_from_address=reduced_data_addr)
    var reduced_indices = IPtr(unsafe_from_address=reduced_indices_addr)
    var reduced_indptr = IPtr(unsafe_from_address=reduced_indptr_addr)
    var old_rows = IPtr(unsafe_from_address=old_rows_addr)

    for cursor in range(nonzeros):
        scratch[Int(row_indices[cursor])] += 1

    var count = 0
    var cursor = Int64(0)
    reduced_indptr[0] = 0
    for row in range(rows):
        var row_count = scratch[row]
        if row_count > 0:
            old_rows[count] = Int64(row)
            scratch[row] = cursor
            cursor += row_count
            count += 1
            reduced_indptr[count] = cursor

    for col in range(cols):
        for source in range(
            Int(col_indptr[col]), Int(col_indptr[col + 1])
        ):
            var row = Int(row_indices[source])
            var target = Int(scratch[row])
            reduced_data[target] = data[source]
            reduced_indices[target] = Int64(col)
            scratch[row] += 1
    return count


@export("mcvx_csc_compact_i32")
def mcvx_csc_compact_i32(
    data_addr: Int,
    row_indices_addr: Int,
    col_indptr_addr: Int,
    rows: Int,
    cols: Int,
    nonzeros: Int,
    scratch_addr: Int,
    reduced_data_addr: Int,
    reduced_indices_addr: Int,
    reduced_indptr_addr: Int,
    old_rows_addr: Int,
) abi("C") -> Int:
    var data = FPtr(unsafe_from_address=data_addr)
    var row_indices = I32Ptr(unsafe_from_address=row_indices_addr)
    var col_indptr = I32Ptr(unsafe_from_address=col_indptr_addr)
    var scratch = I32Ptr(unsafe_from_address=scratch_addr)
    var reduced_data = FPtr(unsafe_from_address=reduced_data_addr)
    var reduced_indices = I32Ptr(unsafe_from_address=reduced_indices_addr)
    var reduced_indptr = I32Ptr(unsafe_from_address=reduced_indptr_addr)
    var old_rows = IPtr(unsafe_from_address=old_rows_addr)

    for cursor in range(nonzeros):
        scratch[Int(row_indices[cursor])] += 1

    var count = 0
    var cursor = Int32(0)
    reduced_indptr[0] = 0
    for row in range(rows):
        var row_count = scratch[row]
        if row_count > 0:
            old_rows[count] = Int64(row)
            scratch[row] = cursor
            cursor += row_count
            count += 1
            reduced_indptr[count] = cursor

    for col in range(cols):
        for source in range(
            Int(col_indptr[col]), Int(col_indptr[col + 1])
        ):
            var row = Int(row_indices[source])
            var target = Int(scratch[row])
            reduced_data[target] = data[source]
            reduced_indices[target] = Int32(col)
            scratch[row] += 1
    return count
