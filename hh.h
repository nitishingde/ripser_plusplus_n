#ifndef PYRIPSER_HH_H
#define PYRIPSER_HH_H

#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

typedef float value_t;
typedef int64_t index_t;

template <class Key, class T>
class hash_map;

typedef struct {
    value_t birth;
    value_t death;
} birth_death_coordinate;

struct diameter_index_t_struct{
    value_t diameter;
    index_t index;
};

class binomial_coeff_table {
    index_t num_n;
    index_t max_tuple_length;

public:
    index_t* binoms;
    binomial_coeff_table(index_t n, index_t k);
    index_t get_num_n() const;
    index_t get_max_tuple_length() const;
    __host__ __device__ index_t operator()(index_t n, index_t k) const;
    ~binomial_coeff_table();
};

class compressed_lower_distance_matrix {
public:
    std::vector<value_t> distances;
    std::vector<value_t*> rows;

    void init_rows() {
        value_t* pointer= &distances[0];
        for (index_t i= 1; i < size(); ++i) {
            rows[i]= pointer;
            pointer+= i;
        }
    }

    compressed_lower_distance_matrix(std::vector<value_t>&& _distances)
            : distances(std::move(_distances)), rows((1 + std::sqrt(1 + 8 * distances.size())) / 2) {
        assert(distances.size() == size() * (size() - 1) / 2);
        init_rows();
    }

    template <typename DistanceMatrix>
    compressed_lower_distance_matrix(const DistanceMatrix& mat)
            : distances(mat.size() * (mat.size() - 1) / 2), rows(mat.size()) {
        init_rows();

        for (index_t i= 1; i < size(); ++i)
            for (index_t j= 0; j < i; ++j) rows[i][j]= mat(i, j);
    }

    value_t operator()(const index_t i, const index_t j) const {
        return i == j ? 0 : i < j ? rows[j][i] : rows[i][j];
    }

    size_t size() const { return rows.size(); }

};

class euclidean_distance_matrix {
public:
    std::vector<std::vector<value_t>> points;

    euclidean_distance_matrix(std::vector<std::vector<value_t>>&& _points)
            : points(std::move(_points)) {
        for (auto p : points) { assert(p.size() == points.front().size()); }
    }

    value_t operator()(const index_t i, const index_t j) const {
        assert(i < points.size());
        assert(j < points.size());
        return std::sqrt(std::inner_product(
                points[i].begin(), points[i].end(), points[j].begin(), value_t(), std::plus<value_t>(),
                [](value_t u, value_t v) { return (u - v) * (u - v); }));
    }

    size_t size() const { return points.size(); }
};

template<typename ValueType>
class compressed_sparse_matrix {
    std::vector<size_t> bounds;
    std::vector<ValueType> entries;

    typedef typename std::vector<ValueType>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

public:
    size_t size() const { return bounds.size(); }

    iterator_pair subrange(const index_t index) {
        return {
            entries.begin() + (index == 0 ? 0 : bounds[index - 1]),
            entries.begin() + bounds[index]
        };
    }

    void append_column() { bounds.push_back(entries.size()); }
};

template<typename ValueType>
class compressed_sparse_submatrix {
    std::vector<size_t> sub_bounds;//the 0-based indices for
    std::vector<ValueType> entries;

    typedef typename std::vector<ValueType>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

public:
    size_t size() const { return sub_bounds.size(); }

    //assume we are given a "subindex" for the submatrix
    //allows iteration from sub_bounds[index_to_subindex[index]] to sub_bounds[index_to_subindex[index+1]]-1
    iterator_pair subrange(const index_t subindex) {
        return {
            entries.begin() + (subindex == 0 ? 0 : sub_bounds[subindex - 1]),
            entries.begin() + sub_bounds[subindex]
        };
    }

    void append_column() { sub_bounds.push_back(entries.size()); }

    void push_back(const ValueType e) {
        assert(0 < size());
        entries.push_back(e);
        ++sub_bounds.back();
    }
};

class ripser {
public:
    compressed_lower_distance_matrix dist;//this can be either sparse or compressed

    index_t n, dim_max;//n is the number of points, dim_max is the max dimension to compute PH
    value_t threshold;//this truncates the filtration by removing simplices too large. low values of threshold should use --sparse option
    float ratio;
    const binomial_coeff_table binomial_coeff;
    mutable std::vector<index_t> vertices;
    mutable std::vector<diameter_index_t_struct> cofacet_entries;
    size_t freeMem, totalMem;
    cudaDeviceProp deviceProp;
    int grid_size;
    //hash_map<index_t, index_t> pivot_column_index;//small hash map for matrix reduction

    //we are removing d_flagarray for a more general array: d_flagarray_OR_index_to_subindex
    //char* type is 3x faster for thrust::count than index_t*
#ifndef ASSEMBLE_REDUCTION_SUBMATRIX
    char* d_flagarray;//an array where d_flagarray[i]= 1 if i satisfies some property and d_flagarray[i]=0 otherwise
#endif
    index_t* h_pivot_column_index_array_OR_nonapparent_cols;//the pivot column index hashmap represented by an array OR the set of nonapparent column indices

    value_t* d_distance_matrix;//GPU copy of the distance matrix

    //d_pivot_column_index_OR_nonapparent_cols is d_nonapparent_cols when used in gpuscan() and compute_pairs() and is d_pivot_column_index when in gpu_assemble_columns()
    index_t* d_pivot_column_index_OR_nonapparent_cols;//the pivot column index hashmap represented on GPU as an array OR the set of nonapparent columns on GPU

    index_t max_num_simplices_forall_dims;//the total number of simplices of dimension dim_max possible (this assumes no threshold condition to sparsify the simplicial complex)
    //the total number of simplices in the dim_max+1 dimension (a factor n larger than max_num_simplices_forall_dims), infeasible to allocate with this number if max_num_simplices_forall_dims is already pushing the memory limits.

    struct diameter_index_t_struct* d_columns_to_reduce;//GPU copy of the columns to reduce depending on the current dimension
    struct diameter_index_t_struct* h_columns_to_reduce;//columns to reduce depending on the current dimension

    binomial_coeff_table* d_binomial_coeff;//GPU copy of the binomial coefficient table
    index_t* h_d_binoms;

    index_t* d_num_columns_to_reduce=nullptr;//use d_num_columns_to_reduce to keep track of the number of columns to reduce
    index_t* h_num_columns_to_reduce;//h_num_columns_to_reduce is tied to d_num_columns_to_reduce in pinned memory?

    index_t* d_num_nonapparent= nullptr;//the number of nonapparent columns. *d_num_columns_to_reduce-*d_num_nonapparent= number of apparent columns
    index_t* h_num_nonapparent;//h_num_nonapparent is tied to d_num_nonapparent in pinned memory?

    index_t num_apparent;//the number of apparent pairs found

    value_t* d_cidx_to_diameter;//GPU side mapping from cidx to diameters for gpuscan faces of a given row of a "lowest one" search

#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)//assemble reduction submatrix
    index_t* d_flagarray_OR_index_to_subindex;//GPU data structure that maps index to subindex
    index_t* h_flagarray_OR_index_to_subindex;//copy of index_to_subindex data structure that acts as a map for matrix index to reduction submatrix indexing on CPU side
#endif

    //for GPU-scan (finding apparent pairs)
    index_t* d_lowest_one_of_apparent_pair;//GPU copy of the lowest ones, d_lowest_one_of_apparent_pair[col]= lowest one row of column col
    //index_t* h_lowest_one_of_apparent_pair;//the lowest ones, d_lowest_one_of_apparent_pair[col]= lowest one row of column col
    struct index_t_pair_struct* d_pivot_array;//sorted array of all pivots, substitute for a structured hashmap with lookup done by log(n) binary search
    struct index_t_pair_struct* h_pivot_array;//sorted array of all pivots
    std::vector<std::vector<birth_death_coordinate>> list_of_barcodes;
public:

    explicit ripser(compressed_lower_distance_matrix&& _dist, index_t _dim_max, value_t _threshold, float _ratio);
    void free_gpumem_dense_computation();
    void free_init_cpumem();
    void free_remaining_cpumem();
    index_t calculate_gpu_dim_max_for_fullrips_computation_from_memory(const index_t dim_max, const bool isfullrips);
    index_t get_num_simplices_for_dim(index_t dim);
    index_t get_next_vertex(index_t& v, const index_t idx, const index_t k) const;
    template <typename OutputIterator>
    OutputIterator get_simplex_vertices(index_t idx, const index_t dim, index_t v, OutputIterator out) const;
    value_t compute_diameter(const index_t index, index_t dim) const;

    class simplex_coboundary_enumerator;

    void gpu_assemble_columns_to_reduce_plusplus(const index_t dim, cudaStream_t cudaStream = 0);
    void cpu_byneighbor_assemble_columns_to_reduce(std::vector<struct diameter_index_t_struct>& simplices, std::vector<struct diameter_index_t_struct>& columns_to_reduce, hash_map<index_t, index_t>& pivot_column_index, index_t dim);
    void assemble_columns_gpu_accel_transition_to_cpu_only(const bool& more_than_one_dim_cpu_only, std::vector<diameter_index_t_struct>& simplices, std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t,index_t>& cpu_pivot_column_index, index_t dim);
    index_t get_value_pivot_array_hashmap(index_t row_cidx, struct row_cidx_column_idx_struct_compare cmp);
    void compute_dim_0_pairs(std::vector<diameter_index_t_struct>& edges, std::vector<diameter_index_t_struct>& columns_to_reduce);
    void gpu_compute_dim_0_pairs(std::vector<struct diameter_index_t_struct>& columns_to_reduce);
    void gpuscan_0(const index_t dim, const index_t num_simplices, cudaStream_t cudaStream);
    void gpuscan_1(const index_t dim, const index_t num_simplices, cudaStream_t cudaStream);
    void gpuscan_2(const index_t dim, const index_t num_simplices, cudaStream_t cudaStream);
    void gpuscan_3(const index_t dim, const index_t num_simplices, cudaStream_t cudaStream);
    void gpuscan_4(const index_t dim, const index_t num_simplices, cudaStream_t cudaStream);
    void gpuscan(const index_t dim);
    template <typename Column>
    diameter_index_t_struct init_coboundary_and_get_pivot_fullmatrix(const diameter_index_t_struct simplex, Column& working_coboundary, const index_t& dim, hash_map<index_t, index_t>& pivot_column_index);
    template <typename Column>
    diameter_index_t_struct init_coboundary_and_get_pivot_submatrix(const diameter_index_t_struct simplex, Column& working_coboundary, index_t dim, struct row_cidx_column_idx_struct_compare cmp);
    template <typename Column>
    void add_simplex_coboundary_oblivious(const diameter_index_t_struct simplex, const index_t& dim, Column& working_coboundary);
    template <typename Column>
    void add_simplex_coboundary_use_reduction_column(const diameter_index_t_struct simplex, const index_t& dim, Column& working_reduction_column, Column& working_coboundary);
    //THIS IS THE METHOD TO CALL FOR CPU SIDE FULL MATRIX REDUCTION
    template <typename Column>
    void add_coboundary_fullmatrix(compressed_sparse_matrix<diameter_index_t_struct>& reduction_matrix, const std::vector<diameter_index_t_struct>& columns_to_reduce, const size_t index_column_to_add, const size_t& dim, Column& working_reduction_column, Column& working_coboundary);
    //THIS IS THE METHOD TO CALL FOR SUBMATRIX REDUCTION ON CPU SIDE
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    template <typename Column>
    void add_coboundary_reduction_submatrix(compressed_sparse_submatrix<diameter_index_t_struct>& reduction_submatrix, const size_t index_column_to_add, const size_t& dim, Column& working_reduction_column, Column& working_coboundary);
#endif
    void compute_pairs(std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t, index_t>& pivot_column_index, index_t dim);
    void compute_pairs_plusplus(index_t dim, index_t gpuscan_startingdim);
    std::vector<diameter_index_t_struct> get_edges();
    void compute_barcodes();
    void init(index_t gpu_dim_max);//@nitish
    void set_h_num_nonapparent(const index_t val);
};

compressed_lower_distance_matrix read_file(std::istream& input_stream);

void print_usage_and_exit(int exit_code);

#endif //PYRIPSER_HH_H
