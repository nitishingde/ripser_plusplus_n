/*
 Ripser++: accelerated Vietoris-Rips persistence barcodes computation with GPU

 MIT License

 Copyright (c) 2019, 2020 Simon Zhang, Mengbai Xiao, Hao Wang

 Python Bindings Contributors:
 Birkan Gokbag
 Ryan DeMilt

 Copyright (c) 2015-2019 Ripser codebase, written by Ulrich Bauer

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or
 upgrades to the features, functionality or performance of the source code
 ("Enhancements") to anyone; however, if you choose to make your Enhancements
 available either publicly, or directly to the author of this software, without
 imposing a separate written license agreement for such Enhancements, then you
 hereby grant the following license: a non-exclusive, royalty-free perpetual
 license to install, use, modify, prepare derivative works, incorporate into
 other computer software, distribute, and sublicense such enhancements or
 derivative works thereof, in binary and source code form.

*/

#define CUDACHECK(cmd) do {\
    cudaError_t e= cmd;\
    if( e != cudaSuccess ) {\
        printf("Failed: Cuda error %s:%d '%s'\n",\
        __FILE__,__LINE__,cudaGetErrorString(e));\
    exit(EXIT_FAILURE);\
    }\
    } while(0)

//#define INDICATE_PROGRESS//DO NOT UNCOMMENT THIS IF YOU WANT TO LOG PROFILING NUMBERS FROM stderr TO FILE
//#define PRINT_PERSISTENCE_PAIRS//print out all persistence paris to stdout
//#define CPUONLY_ASSEMBLE_REDUCTION_MATRIX//do full matrix reduction on CPU with the sparse coefficient matrix V
#define ASSEMBLE_REDUCTION_SUBMATRIX//do submatrix reduction with the sparse coefficient submatrix of V
//#define PROFILING
//#define COUNTING
#define USE_PHASHMAP//www.github.com/greg7mdp/parallel-hashmap
#define PYTHON_BARCODE_COLLECTION
#ifndef USE_PHASHMAP
#define USE_GOOGLE_HASHMAP
#endif

//#define CPUONLY_SPARSE_HASHMAP//WARNING: MAY NEED LOWER GCC VERSION TO RUN, TESTED ON: NVCC VERSION 9.2 WITH GCC VERSIONS >=5.3.0 AND <=7.3.0

#define MIN_INT64 (-9223372036854775807-1)
#define MAX_INT64 (9223372036854775807)
#define MAX_FLOAT (340282346638528859811704183484516925440.000000)

#include "hh.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <profiling/stopwatch.h>
#include <sparsehash/dense_hash_map>
#include <phmap_interface/phmap_interface.h>

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#ifdef CPUONLY_SPARSE_HASHMAP
#include <sparsehash/sparse_hash_map>
template <class Key, class T> class hash_map : public google::sparse_hash_map<Key, T> {
public:
    explicit hash_map() : google::sparse_hash_map<Key, T>() {
        }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifndef CPUONLY_SPARSE_HASHMAP
template <class Key, class T> class hash_map : public google::dense_hash_map<Key, T> {
public:
    explicit hash_map() : google::dense_hash_map<Key, T>() {
        this->set_empty_key(-1);
    }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifdef INDICATE_PROGRESS
static const std::chrono::milliseconds time_step(40);
static const std::string clear_line("\r\033[K");
#endif

struct greaterdiam_lowerindex_diameter_index_t_struct_compare {
    __host__ __device__ bool operator() (struct diameter_index_t_struct a, struct diameter_index_t_struct b){
        return a.diameter!=b.diameter ? a.diameter>b.diameter : a.index<b.index;
    }
};
struct greaterdiam_lowerindex_diameter_index_t_struct_compare_reverse {
    __host__ __device__ bool operator() (struct diameter_index_t_struct a, struct diameter_index_t_struct b){
        return a.diameter!=b.diameter ? a.diameter<b.diameter : a.index>b.index;
    }
};

struct index_t_pair_struct{//data type for a pivot in the coboundary matrix: (row,column)
    index_t row_cidx;
    index_t column_idx;
};

typedef struct{
    index_t num_barcodes;
    birth_death_coordinate* barcodes;
} set_of_barcodes;

typedef struct{
    int num_dimensions;
    set_of_barcodes* all_barcodes;
} ripser_plusplus_result;

struct row_cidx_column_idx_struct_compare{
    __host__ __device__ bool operator()(struct index_t_pair_struct a, struct index_t_pair_struct b){
        //return a.row_cidx!=b.row_cidx ? a.row_cidx<b.row_cidx : a.column_idx<b.column_idx;//the second condition should never happen if sorting pivot pairs since pivots do not conflict on rows or columns
        return a.row_cidx<b.row_cidx || (a.row_cidx==b.row_cidx && a.column_idx<b.column_idx);
    }
};

__host__ __device__ value_t hd_max(value_t a, value_t b){
    return a>b?a:b;
}

void check_overflow(index_t i){
    if(i<0){
        throw std::overflow_error("simplex index "+std::to_string((uint64_t)i)+" in filtration is overflowing past 64 bits signed integer");
    }
}

//assume i>j (lower triangular with i indexing rows and j indexing columns
#define LOWER_DISTANCE_INDEX(i,j,n) (((i)*((i)-1)/2)+(j))

#define BINOM_TRANSPOSE(i,j) ((j)*(num_n)+(i))

binomial_coeff_table::binomial_coeff_table(index_t n, index_t k) {
    binoms= (index_t*)malloc(sizeof(index_t)*(n+1)*(k+1));
    if(binoms==nullptr){
        std::cerr<<"malloc for binoms failed"<<std::endl;
        exit(1);
    }
    num_n= n+1;
    max_tuple_length= k+1;
    memset(binoms, 0, sizeof(index_t)*num_n*max_tuple_length);

    for(index_t i= 0; i <= n; i++) {
        for(index_t j= 0; j <= std::min(i, k); j++) {
            if(j == 0 || j == i) {
                binoms[BINOM_TRANSPOSE(i,j)]= 1;
            }
            else {
                binoms[BINOM_TRANSPOSE(i,j)]= binoms[BINOM_TRANSPOSE(i-1,j-1)]+binoms[BINOM_TRANSPOSE(i-1,j)];
            }
        }
        check_overflow(binoms[BINOM_TRANSPOSE(i,std::min(i>>1,k))]);
    }
}

index_t binomial_coeff_table::get_num_n() const{
    return num_n;
}

index_t binomial_coeff_table::get_max_tuple_length() const {
    return max_tuple_length;
}

__host__ __device__ index_t binomial_coeff_table::operator()(index_t n, index_t k) const {
    assert(n<num_n && k<max_tuple_length);
    return binoms[BINOM_TRANSPOSE(n,k)];
}

binomial_coeff_table::~binomial_coeff_table(){
    free(binoms);
}

class union_find {
    std::vector<index_t> parent;
    std::vector<uint8_t> rank;

public:
    union_find(index_t n) : parent(n), rank(n, 0) {
        for (index_t i= 0; i < n; ++i) parent[i]= i;
    }

    index_t find(index_t x) {
        index_t y= x, z;
        while ((z= parent[y]) != y) y= z;
        while ((z= parent[x]) != y) {
            parent[x]= y;
            x= z;
        }
        return z;
    }
    void link(index_t x, index_t y) {
        if ((x= find(x)) == (y= find(y))) return;
        if (rank[x] > rank[y])
            parent[y]= x;
        else {
            parent[x]= y;
            if (rank[x] == rank[y]) ++rank[y];
        }
    }
};

template <typename Heap> struct diameter_index_t_struct pop_pivot(Heap& column) {
    if(column.empty()) {
        return {0,-1};
    }

    auto pivot= column.top();
    column.pop();
    while(!column.empty() && (column.top()).index == pivot.index) {
        column.pop();
        if (column.empty()) {
            return {0,-1};
        }
        else {
            pivot= column.top();
            column.pop();
        }
    }
    return pivot;
}

template <typename Heap> struct diameter_index_t_struct get_pivot(Heap& column) {
    struct diameter_index_t_struct result= pop_pivot(column);
    if (result.index != -1) column.push(result);
    return result;
}

template <typename T> T begin(std::pair<T, T>& p) { return p.first; }
template <typename T> T end(std::pair<T, T>& p) { return p.second; }

template <class Predicate> index_t upper_bound(index_t top, Predicate pred) {
    if (!pred(top)) {
        index_t count= top;
        while (count > 0) {
            index_t step= count >> 1;
            if (!pred(top - step)) {
                top-= step + 1;
                count-= step + 1;
            } else
                count= step;
        }
    }
    return top;
}

__global__ void gpu_insert_pivots_kernel(struct index_t_pair_struct* d_pivot_array, index_t* d_lowest_one_of_apparent_pair, index_t* d_pivot_column_index_OR_nonapparent_cols, index_t num_columns_to_reduce, index_t* d_num_nonapparent){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x*(index_t)gridDim.x;

    //index_t* d_pivot_column_index_OR_nonapparent_cols is being used as d_nonapparent_cols
    for(; tid<num_columns_to_reduce; tid+= stride) {
        int keep_tid= d_lowest_one_of_apparent_pair[tid] == -1;
        if (!keep_tid) {//insert pivot
            d_pivot_array[tid].row_cidx= d_lowest_one_of_apparent_pair[tid];
            d_pivot_array[tid].column_idx= tid;
        }else {//keep track of nonapparent columns
            d_pivot_array[tid].row_cidx= MAX_INT64;
            d_pivot_array[tid].column_idx= MAX_INT64;

            //do standard warp based filtering under the assumption that there are few nonapparent columns
#define FULL_MASK 0xFFFFFFFF
            int lane_id= threadIdx.x % 32;
            int mask= __ballot_sync(FULL_MASK, keep_tid);
            int leader= __ffs(mask) - 1;
            int base;
            if (lane_id == leader)
                base= atomicAdd((unsigned long long int *) d_num_nonapparent, __popc(mask));
            base= __shfl_sync(mask, base, leader);
            int pos= base + __popc(mask & ((1 << lane_id) - 1));

            if (keep_tid) {
                d_pivot_column_index_OR_nonapparent_cols[pos]= tid;//being used as d_nonapparent_cols
            }
        }
    }
}

template<typename T> __global__ void populate_edges(T* d_flagarray, struct diameter_index_t_struct* d_columns_to_reduce, value_t threshold, value_t* d_distance_matrix, index_t max_num_simplices, index_t num_points, binomial_coeff_table* d_binomial_coeff){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x*(index_t)gridDim.x;

    __shared__ index_t shared_vertices[256][3];//designed to eliminate bank conflicts (that's what the 3 is for)
    for(; tid<max_num_simplices; tid+= stride) {
        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= 2; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x][offset++]= v;

            idx-= (*d_binomial_coeff)(v, k);
        }
        //shared_vertices is sorted in decreasing order
        value_t diam= d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x][0], shared_vertices[threadIdx.x][1], num_points)];
        if(diam<=threshold){
            d_columns_to_reduce[tid].diameter= diam;
            d_columns_to_reduce[tid].index= tid;
            d_flagarray[tid]= 1;
        }else{
            d_columns_to_reduce[tid].diameter= MAX_FLOAT;//the sorting is in boundary matrix filtration order
            d_columns_to_reduce[tid].index= MIN_INT64;
            d_flagarray[tid]= 0;
        }
    }
}
//the hope is that this is concurrency-bug free, however this is very bad for sparse graph performance

template<typename T> __global__ void populate_columns_to_reduce(T* d_flagarray, struct diameter_index_t_struct* d_columns_to_reduce, index_t* d_pivot_column_index,
                                                                value_t* d_distance_matrix, index_t num_points, index_t max_num_simplices, index_t dim, value_t threshold, binomial_coeff_table* d_binomial_coeff) {
    index_t tid= (index_t)threadIdx.x + (index_t)blockIdx.x * (index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block
    for (; tid < max_num_simplices; tid+= stride) {

        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;
            idx-= (*d_binomial_coeff)(v, k);
        }

        value_t diam= -MAX_FLOAT;

        for(index_t i= 0; i<=dim; i++){
            for(index_t j= i+1; j<=dim; j++){
                diam= hd_max(diam, d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x * (dim + 1) + i], shared_vertices[threadIdx.x * (dim + 1) + j], num_points)]);
            }
        }

        if(d_pivot_column_index[tid]==-1 && diam<=threshold){
            d_columns_to_reduce[tid].diameter= diam;
            d_columns_to_reduce[tid].index= tid;
            d_flagarray[tid]= 1;
        }else{
            d_columns_to_reduce[tid].diameter= -MAX_FLOAT;
            d_columns_to_reduce[tid].index= MAX_INT64;
            d_flagarray[tid]= 0;
        }
    }
}

__global__ void init_cidx_to_diam(value_t* d_cidx_to_diameter, struct diameter_index_t_struct* d_columns_to_reduce, index_t num_columns_to_reduce){
    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    for (; tid < num_columns_to_reduce; tid += stride) {
        d_cidx_to_diameter[d_columns_to_reduce[tid].index]= d_columns_to_reduce[tid].diameter;
    }
}

//scatter operation
__global__ void init_index_to_subindex(index_t* d_index_to_subindex, index_t* d_nonapparent_columns, index_t num_nonapparent){
    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    for (; tid < num_nonapparent; tid += stride) {
        d_index_to_subindex[d_nonapparent_columns[tid]]= tid;
    }
}

//THIS IS THE GPU SCAN KERNEL for the dense case!!
__global__ void coboundary_findapparent_single_kernel(value_t* d_cidx_to_diameter, struct diameter_index_t_struct * d_columns_to_reduce, index_t* d_lowest_one_of_apparent_pair,  const index_t dim, index_t num_simplices, const index_t num_points, binomial_coeff_table* d_binomial_coeff, index_t num_columns_to_reduce, value_t* d_distance_matrix, value_t threshold) {

    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block

    for (; tid < num_columns_to_reduce; tid += stride) {

        //populate the shared_vertices[][] matrix with vertex indices of the column index= shared_vertices[threadIdx.x][-];
        //shared_vertices[][] matrix has row index threadIdx.x and col index offset, represented by: shared_vertices[threadIdx.x * (dim + 1) + offset]=
        index_t offset= 0;

        index_t v= num_points - 1;
        index_t idx= d_columns_to_reduce[tid].index;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;//set v to the largest possible vertex index given idx as a combinatorial index

            idx-= (*d_binomial_coeff)(v, k);
        }
        v= num_points-1;//this keeps track of the newly added vertex to the set of vertices stored in shared_vertices[threadIdx.x][-] to form a cofacet of the columns
        index_t k= dim+1;
        index_t idx_below= d_columns_to_reduce[tid].index;
        index_t idx_above= 0;
        while ((v != -1) && ((*d_binomial_coeff)(v, k) <= idx_below)) {
            idx_below -= (*d_binomial_coeff)(v, k);
            idx_above += (*d_binomial_coeff)(v, k + 1);
            --v;
            --k;
            assert(k != -1);
        }
        while(v!=-1) {//need to enumerate cofacet combinatorial index in reverse lexicographic order (largest cidx down to lowest cidx)
            index_t row_combinatorial_index= idx_above + (*d_binomial_coeff)(v--, k + 1) + idx_below;

            //find the cofacet diameter
            value_t cofacet_diam= d_columns_to_reduce[tid].diameter;
            for(index_t j=0; j<dim+1; j++){
                index_t last_v= v+1;
                index_t simplex_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                if(last_v>simplex_v){
                    cofacet_diam= hd_max(cofacet_diam, d_distance_matrix[LOWER_DISTANCE_INDEX(last_v, shared_vertices[threadIdx.x * (dim + 1) + j], num_points)]);
                }else{
                    cofacet_diam= hd_max(cofacet_diam, d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x * (dim + 1) + j], last_v, num_points)]);
                }
            }
            if(d_columns_to_reduce[tid].diameter==cofacet_diam) {//this is a sufficient condition to finding a lowest one

                //check if there is a nonzero to the left of (row_combinatorial_index, tid) in the coboundary matrix

                //extra_vertex is the "added" vertex to shared_vertices
                //FACT: {shared_vertices[threadIdx.x*(dim+1)+0]... threadIdx.x*(dim+1)+dim] union extra_vertex} equals cofacet vertices

                index_t prev_remove_v= -1;
                index_t s_v= shared_vertices[threadIdx.x * (dim + 1)];//the largest indexed vertex, shared_vertices is sorted in decreasing orders
                bool passed_extra_v= false;
                index_t remove_v;//this is the vertex to remove from the cofacet
                index_t extra_vertex= v+1;//the +1 is here to counteract the last v-- line of code
                if(s_v>extra_vertex){
                    remove_v= s_v;
                }else{
                    remove_v= extra_vertex;
                    passed_extra_v= true;
                }
                prev_remove_v= remove_v;

                index_t facet_of_row_combinatorial_index= row_combinatorial_index;
                facet_of_row_combinatorial_index-= (*d_binomial_coeff)(remove_v, dim+2);//subtract the largest binomial coefficient to get the new cidx

                index_t col_cidx= d_columns_to_reduce[tid].index;
                value_t facet_of_row_diameter= d_cidx_to_diameter[facet_of_row_combinatorial_index];
                value_t col_diameter= d_columns_to_reduce[tid].diameter;

                if(facet_of_row_combinatorial_index==col_cidx && facet_of_row_diameter== col_diameter){//if there is an exact match of the tid column and the face of the row, then all subsequent faces to search will be to the right of column tid
                    //coboundary column tid has an apparent pair, record it
                    d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                    break;
                }

                    //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<= threshold && (
                    //        d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                    //        || (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                    //        || facet_of_row_combinatorial_index> d_columns_to_reduce[tid].index)){
                    //FACT: it turns out we actually only need to check facet_of_row_diameter<= threshold &&(facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx)
                    //since we should never have a facet of the cofacet with diameter larger than the cofacet's diameter= column's diameter
                    //in fact, we don't even need to check facet_of_row_diameter<=threshold since diam(face(cofacet(simplex)))<=diam(cofacet(simplex))=diam(simplex)<=threshold
                    //furthermore, we don't even need to check facet_of_row_combinatorial_index<col_cidx since we will exit upon col_cidx while iterating in increasing combinatorial index
                else if(facet_of_row_diameter==col_diameter){
                    assert(facet_of_row_diameter<= threshold && (facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx));
                    d_lowest_one_of_apparent_pair[tid]= -1;
                    break;
                }
                bool found_apparent_or_found_nonzero_to_left= false;

                //need to remove the last vertex: extra_v during searches
                //there are dim+2 total number of vertices, the largest vertex was already checked so that is why k starts at dim+1
                //j is the col. index e.g. shared_vertices[threadIdx.x][j]=shared_vertices[threadIdx.x*(dim+1)+j]
                for(index_t k= dim+1, j=passed_extra_v?0:1; k>=1; k--){//start the loop after checking the lexicographically smallest facet boundary case
                    if(passed_extra_v) {
                        remove_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        j++;
                    }
                    else if(j<dim+1) {
                        //compare s_v in shared_vertices with v
                        index_t s_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        if (s_v > extra_vertex) {
                            remove_v= s_v;
                            j++;
                        } else {
                            remove_v= extra_vertex;//recall: extra_vertex= v+1
                            passed_extra_v= true;
                        }
                        //this last else says: if j==dim+1 and we never passed extra vertex, then we must remove extra_vertex as the last vertex to remove to form a facet.
                    }else {//there is no need to check s_v>extra_vertex, we never passed extra_vertex, so we need to remove extra_vertex for the last check
                        remove_v= extra_vertex;//recall; v+1 since there is a v-- before this
                        passed_extra_v= true;
                    }

                    //exchange remove_v choose k with prev_remove_v choose k
                    facet_of_row_combinatorial_index-=(*d_binomial_coeff)(remove_v,k);
                    facet_of_row_combinatorial_index+= (*d_binomial_coeff)(prev_remove_v,k);

                    value_t facet_of_row_diameter= d_cidx_to_diameter[facet_of_row_combinatorial_index];

                    if(facet_of_row_combinatorial_index==col_cidx && facet_of_row_diameter==col_diameter){
                        //coboundary column tid has an apparent pair, record it
                        d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;///need to break out the while(v!=-1) loop
                    }

                        //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<=threshold &&
                        //( d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                        //|| (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                        //|| facet_of_row_combinatorial_index>d_columns_to_reduce[tid].index)){
                    else if(facet_of_row_diameter==col_diameter){
                        assert(facet_of_row_diameter<= threshold && (facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx));
                        //d_lowest_one_of_apparent_pair[] is set to -1's already though...
                        d_lowest_one_of_apparent_pair[tid]= -1;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;
                    }

                    prev_remove_v= remove_v;
                }
                //we must exit early if we have a nonzero to left or the column is apparent
                if(found_apparent_or_found_nonzero_to_left){
                    break;
                }


                //end check for nonzero to left

                //need to record the found pairs in the global hash_map for pairs (post processing)
                //see post processing section in gpuscan method
            }
            while ((v != -1) && ((*d_binomial_coeff)(v, k) <= idx_below)) {
                idx_below -= (*d_binomial_coeff)(v, k);
                idx_above += (*d_binomial_coeff)(v, k + 1);
                --v;
                --k;
                assert(k != -1);
            }
        }
    }
}

class ripser::simplex_coboundary_enumerator {
private:
    index_t idx_below, idx_above, v, k;
    std::vector<index_t> vertices;
    ///const diameter_index_t simplex;
    const struct diameter_index_t_struct simplex;
    const compressed_lower_distance_matrix& dist;
    const binomial_coeff_table& binomial_coeff;

public:

    simplex_coboundary_enumerator(
            const struct diameter_index_t_struct _simplex, index_t _dim,
            const ripser& parent)
            : idx_below(_simplex.index),
              idx_above(0), v(parent.n - 1), k(_dim + 1),
              vertices(_dim + 1), simplex(_simplex), dist(parent.dist),
              binomial_coeff(parent.binomial_coeff) {
        parent.get_simplex_vertices(_simplex.index, _dim, parent.n, vertices.begin());

    }

    bool has_next(bool all_cofacets= true) {
        return (v >= k && (all_cofacets || binomial_coeff(v, k) > idx_below));//second condition after the || is to ensure iteration of cofacets with no need to adjust
    }

    struct diameter_index_t_struct next() {
        while ((binomial_coeff(v, k) <= idx_below)) {
            idx_below -= binomial_coeff(v, k);
            idx_above += binomial_coeff(v, k + 1);
            --v;
            --k;
            assert(k != -1);
        }
        value_t cofacet_diameter= simplex.diameter;
        for (index_t w : vertices) cofacet_diameter= std::max(cofacet_diameter, dist(v, w));
        index_t cofacet_index= idx_above + binomial_coeff(v--, k + 1) + idx_below;
        return {cofacet_diameter, cofacet_index};
    }
};

ripser::ripser(compressed_lower_distance_matrix&& _dist, index_t _dim_max, value_t _threshold, float _ratio)
    : dist(std::move(_dist)), n(dist.size()), dim_max(std::min(_dim_max, index_t(dist.size() - 2))), threshold(_threshold), ratio(_ratio), binomial_coeff(n, dim_max + 2) {
    for(index_t i = 0; i <= dim_max; i++) {
        list_of_barcodes.push_back(std::vector<birth_death_coordinate>());
    }
}

void ripser::free_gpumem_dense_computation() {
    if (n>=10) {//this fixes a bug for single point persistence being called repeatedly
        cudaFree(d_columns_to_reduce);
#ifndef ASSEMBLE_REDUCTION_SUBMATRIX
        cudaFree(d_flagarray);
#endif
        cudaFree(d_cidx_to_diameter);
//            if (n >= 10) {
        cudaFree(d_distance_matrix);
//            }
        cudaFree(d_pivot_column_index_OR_nonapparent_cols);
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
        cudaFree(d_flagarray_OR_index_to_subindex);
#endif
//            if (binomial_coeff.get_num_n() * binomial_coeff.get_max_tuple_length() > 0) {
        cudaFree(h_d_binoms);
//            }
        cudaFree(d_binomial_coeff);
        cudaFree(d_lowest_one_of_apparent_pair);
        cudaFree(d_pivot_array);
    }
}

void ripser::free_init_cpumem() {
    free(h_pivot_column_index_array_OR_nonapparent_cols);
}

void ripser::free_remaining_cpumem(){
    free(h_columns_to_reduce);
    free(h_pivot_array);
    //pivot_column_index.resize(0);
}

index_t ripser::calculate_gpu_dim_max_for_fullrips_computation_from_memory(const index_t dim_max, const bool isfullrips){

    if(dim_max==0)return 0;
    index_t gpu_dim_max= dim_max;
    index_t gpu_alloc_memory_in_bytes= 0;
    cudaGetDeviceProperties(&deviceProp, 0);

    cudaMemGetInfo(&freeMem,&totalMem);
#ifdef PROFILING
    std::cerr<<"GPU memory before full rips memory calculation, total mem: "<< totalMem<<" bytes, free mem: "<<freeMem<<" bytes"<<std::endl;
#endif
    do{
        index_t gpu_num_simplices_forall_dims= gpu_dim_max<n/2?get_num_simplices_for_dim(gpu_dim_max): get_num_simplices_for_dim(n/2);
        index_t gpumem_char_array_bytes= sizeof(char)*gpu_num_simplices_forall_dims;
        index_t gpumem_index_t_array_bytes= sizeof(index_t)*gpu_num_simplices_forall_dims;
        index_t gpumem_value_t_array_bytes= sizeof(value_t)*gpu_num_simplices_forall_dims;
        index_t gpumem_index_t_pairs_array_bytes= sizeof(index_t_pair_struct)*gpu_num_simplices_forall_dims;
        index_t gpumem_diameter_index_t_array_bytes= sizeof(diameter_index_t_struct)*gpu_num_simplices_forall_dims;
        index_t gpumem_dist_matrix_bytes= sizeof(value_t)*(n*(n-1))/2;
        index_t gpumem_binomial_coeff_table_bytes= sizeof(index_t)*binomial_coeff.get_num_n()*binomial_coeff.get_max_tuple_length() +sizeof(binomial_coeff_table);
        index_t gpumem_index_t_bytes= sizeof(index_t);
        //gpumem_CSR_dist_matrix_bytes is estimated to have n*(n-1)/2 number of nonzeros as an upper bound
        index_t gpumem_CSR_dist_matrix_bytes= sizeof(index_t)*(n+1+4)+(sizeof(index_t)+sizeof(value_t))*n*(n-1)/2;//dist.num_entries;//sizeof(value_t)*(n*(n-1))/2;

        if(isfullrips) {//count the allocated memory for dense case
            gpu_alloc_memory_in_bytes= gpumem_diameter_index_t_array_bytes +
                                       #ifndef ASSEMBLE_REDUCTION_SUBMATRIX
                                       gpumem_char_array_bytes +
                                       #endif
                                       gpumem_value_t_array_bytes +
                                       #ifdef ASSEMBLE_REDUCTION_SUBMATRIX
                                       gpumem_index_t_array_bytes+
                                       #endif
                                       gpumem_dist_matrix_bytes +
                                       gpumem_index_t_array_bytes +
                                       gpumem_binomial_coeff_table_bytes +
                                       gpumem_index_t_bytes * 2 +
                                       gpumem_index_t_array_bytes +
                                       gpumem_index_t_pairs_array_bytes +
                                       gpumem_index_t_pairs_array_bytes;//this last one is for thrust radix sorting buffer

#ifdef PROFILING
            //std::cerr<<"free gpu memory for full rips by calculation in bytes for gpu dim: "<<gpu_dim_max<<": "<<freeMem-gpu_alloc_memory_in_bytes<<std::endl;
                std::cerr<<"gpu memory needed for full rips by calculation in bytes for dim: "<<gpu_dim_max<<": "<<gpu_alloc_memory_in_bytes<<" bytes"<<std::endl;
#endif
            if (gpu_alloc_memory_in_bytes <= freeMem){
                return gpu_dim_max;
            }
        }else{//count the alloced memory for sparse case
            //includes the d_simplices array used in sparse computation for an approximation for both sparse and full rips compelexes?
            gpu_alloc_memory_in_bytes= gpumem_diameter_index_t_array_bytes
                                       #ifdef ASSEMBlE_REDUCTION_SUBMATRIX
                                       + gpumem_index_t_array_bytes
                                       #endif
                                       + gpumem_CSR_dist_matrix_bytes
                                       + gpumem_diameter_index_t_array_bytes
                                       + gpumem_index_t_array_bytes
                                       + gpumem_binomial_coeff_table_bytes
                                       + gpumem_index_t_array_bytes
                                       + gpumem_index_t_pairs_array_bytes
                                       + gpumem_index_t_bytes*4
                                       + gpumem_diameter_index_t_array_bytes
                                       + gpumem_index_t_pairs_array_bytes;//last one is for buffer needed for sorting
#ifdef PROFILING
            //std::cerr<<"(sparse) free gpu memory for full rips by calculation in bytes for gpu dim: "<<gpu_dim_max<<": "<<freeMem-gpu_alloc_memory_in_bytes<<std::endl;
                std::cerr<<"(sparse) gpu memory needed for full rips by calculation in bytes for dim: "<<gpu_dim_max<<": "<<gpu_alloc_memory_in_bytes<<" bytes"<<std::endl;
#endif
            if (gpu_alloc_memory_in_bytes <= freeMem){
                return gpu_dim_max;
            }
        }
        gpu_dim_max--;
    }while(gpu_dim_max>=0);
    return 0;
}

index_t ripser::get_num_simplices_for_dim(index_t dim){
    //beware if dim+1>n and where dim is negative
    assert(dim+1<=n && dim+1>=0);
    return binomial_coeff(n, dim + 1);
}

index_t ripser::get_next_vertex(index_t& v, const index_t idx, const index_t k) const {
    return v= upper_bound(
            v, [&](const index_t& w) -> bool { return (binomial_coeff(w, k) <= idx); });
}

template <typename OutputIterator>
OutputIterator ripser::get_simplex_vertices(index_t idx, const index_t dim, index_t v, OutputIterator out) const {
    --v;
    for (index_t k= dim + 1; k > 0; --k) {
        get_next_vertex(v, idx, k);
        *out++= v;
        idx-= binomial_coeff(v, k);
    }
    return out;
}

value_t ripser::compute_diameter(const index_t index, index_t dim) const {
    value_t diam= -std::numeric_limits<value_t>::infinity();

    vertices.clear();
    get_simplex_vertices(index, dim, dist.size(), std::back_inserter(vertices));

    for (index_t i= 0; i <= dim; ++i)
        for (index_t j= 0; j < i; ++j) {
            diam= std::max(diam, dist(vertices[i], vertices[j]));
        }
    return diam;
}

template <typename Column>
diameter_index_t_struct ripser::init_coboundary_and_get_pivot_fullmatrix(const diameter_index_t_struct simplex, Column& working_coboundary, const index_t& dim, hash_map<index_t, index_t>& pivot_column_index) {
    bool check_for_emergent_pair= true;
    cofacet_entries.clear();
    ripser::simplex_coboundary_enumerator cofacets(simplex, dim, *this);
    while (cofacets.has_next()) {
        diameter_index_t_struct cofacet= cofacets.next();
        if (cofacet.diameter <= threshold) {
            cofacet_entries.push_back(cofacet);
            if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                if (pivot_column_index.find(cofacet.index) == pivot_column_index.end()){
                    return cofacet;
                }
                check_for_emergent_pair= false;
            }
        }
    }
    for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
    return get_pivot(working_coboundary);
}

template <typename Column>
diameter_index_t_struct ripser::init_coboundary_and_get_pivot_submatrix(const diameter_index_t_struct simplex, Column& working_coboundary, index_t dim, struct row_cidx_column_idx_struct_compare cmp) {
    bool check_for_emergent_pair= true;
    cofacet_entries.clear();
    ripser::simplex_coboundary_enumerator cofacets(simplex, dim, *this);
    while (cofacets.has_next()) {
        diameter_index_t_struct cofacet= cofacets.next();
        if (cofacet.diameter <= threshold) {
            cofacet_entries.push_back(cofacet);
            if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                if(get_value_pivot_array_hashmap(cofacet.index, cmp)==-1) {
                    return cofacet;
                }
                check_for_emergent_pair= false;
            }
        }
    }
    for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
    return get_pivot(working_coboundary);
}

template <typename Column>
void ripser::add_simplex_coboundary_oblivious(const diameter_index_t_struct simplex, const index_t& dim, Column& working_coboundary) {
    ripser::simplex_coboundary_enumerator cofacets(simplex, dim, *this);
    while (cofacets.has_next()) {
        diameter_index_t_struct cofacet= cofacets.next();
        if (cofacet.diameter <= threshold) working_coboundary.push(cofacet);
    }
}

template <typename Column>
void ripser::add_simplex_coboundary_use_reduction_column(const diameter_index_t_struct simplex, const index_t& dim, Column& working_reduction_column, Column& working_coboundary) {
    working_reduction_column.push(simplex);
    ripser::simplex_coboundary_enumerator cofacets(simplex, dim, *this);
    while (cofacets.has_next()) {
        diameter_index_t_struct cofacet= cofacets.next();
        if (cofacet.diameter <= threshold) working_coboundary.push(cofacet);
    }
}

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
template <typename Column>
void ripser::add_coboundary_reduction_submatrix(compressed_sparse_submatrix<diameter_index_t_struct>& reduction_submatrix, const size_t index_column_to_add, const size_t& dim, Column& working_reduction_column, Column& working_coboundary) {
    diameter_index_t_struct column_to_add= h_columns_to_reduce[index_column_to_add];
    add_simplex_coboundary_use_reduction_column(column_to_add, dim, working_reduction_column, working_coboundary);
    index_t subindex= h_flagarray_OR_index_to_subindex[index_column_to_add];//this is only defined when ASSEMBLE_REDUCTION_SUBMATRIX is defined
    if(subindex>-1) {
        for (diameter_index_t_struct simplex : reduction_submatrix.subrange(subindex)) {
            add_simplex_coboundary_use_reduction_column(simplex, dim, working_reduction_column, working_coboundary);
        }
    }
}
#endif

void ripser::compute_pairs(std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t, index_t>& pivot_column_index, index_t dim) {
    std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
    for(index_t index_column_to_reduce= 0; index_column_to_reduce < columns_to_reduce.size(); ++index_column_to_reduce) {
        auto column_to_reduce= columns_to_reduce[index_column_to_reduce];
        std::priority_queue<diameter_index_t_struct, std::vector<diameter_index_t_struct>,
                greaterdiam_lowerindex_diameter_index_t_struct_compare> working_coboundary;
        value_t diameter = column_to_reduce.diameter;
        index_t index_column_to_add = index_column_to_reduce;
        diameter_index_t_struct pivot;
        // initialize index bounds of reduction matrix
        pivot = init_coboundary_and_get_pivot_fullmatrix(columns_to_reduce[index_column_to_add], working_coboundary, dim, pivot_column_index);
        while(true) {
            if(pivot.index!=-1) {
                auto left_pair= pivot_column_index.find(pivot.index);
                if(left_pair != pivot_column_index.end()) {
                    index_column_to_add= left_pair->second;
                    add_simplex_coboundary_oblivious(columns_to_reduce[index_column_to_add], dim, working_coboundary);
                    pivot= get_pivot(working_coboundary);
                }
                else {
                    value_t death= pivot.diameter;
                    if(death > diameter * ratio) {
                        std::cout << " [" << diameter << "," << death << ")" << std::endl
                                  << std::flush;
                        birth_death_coordinate barcode = {diameter,death};
                        list_of_barcodes[dim].push_back(barcode);
                    }
                    pivot_column_index[pivot.index]= index_column_to_reduce;

                    break;
                }
            }
            else {
                std::cout << " [" << diameter << ", )" << std::endl << std::flush;
                break;
            }
        }
    }
}

void ripser::compute_pairs_plusplus(index_t dim, index_t gpuscan_startingdim) {

    std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
    compressed_sparse_submatrix<diameter_index_t_struct> reduction_submatrix;
    struct row_cidx_column_idx_struct_compare cmp_pivots;
    index_t num_columns_to_iterate= *h_num_columns_to_reduce;
    if(dim>=gpuscan_startingdim){
        num_columns_to_iterate= *h_num_nonapparent;
    }
    for (index_t sub_index_column_to_reduce= 0; sub_index_column_to_reduce < num_columns_to_iterate;
         ++sub_index_column_to_reduce) {
        index_t index_column_to_reduce =sub_index_column_to_reduce;
        if(dim>=gpuscan_startingdim) {
            index_column_to_reduce= h_pivot_column_index_array_OR_nonapparent_cols[sub_index_column_to_reduce];//h_nonapparent_cols
        }
        auto column_to_reduce= h_columns_to_reduce[index_column_to_reduce];

        std::priority_queue<diameter_index_t_struct, std::vector<diameter_index_t_struct>,
                greaterdiam_lowerindex_diameter_index_t_struct_compare>
                working_reduction_column,
                working_coboundary;

        value_t diameter= column_to_reduce.diameter;

        index_t index_column_to_add= index_column_to_reduce;

        struct diameter_index_t_struct pivot;
        reduction_submatrix.append_column();
        pivot= init_coboundary_and_get_pivot_submatrix(column_to_reduce, working_coboundary, dim, cmp_pivots);

        while (true) {
            if(pivot.index!=-1) {
                index_column_to_add= get_value_pivot_array_hashmap(pivot.index,cmp_pivots);
                if(index_column_to_add!=-1) {
                    add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add, dim, working_reduction_column, working_coboundary);
                    pivot= get_pivot(working_coboundary);
                }
                else{
                    value_t death= pivot.diameter;
                    if (death > diameter * ratio) {
                        std::cout << " [" << diameter << "," << death << ")" << std::endl
                                  << std::flush;
                        birth_death_coordinate barcode = {diameter,death};
                        list_of_barcodes[dim].push_back(barcode);
                    }

                    phmap_put(pivot.index, index_column_to_reduce);
                    while (true) {
                        diameter_index_t_struct e= pop_pivot(working_reduction_column);
                        if (e.index == -1) break;
                        reduction_submatrix.push_back(e);
                    }
                    break;
                }
            }
            else {
                std::cout << " [" << diameter << ", )" << std::endl << std::flush;
                break;
            }
        }
    }
}

std::vector<diameter_index_t_struct> ripser::get_edges() {
    std::vector<diameter_index_t_struct> edges;
    for (index_t index= binomial_coeff(n, 2); index-- > 0;) {
        value_t diameter= compute_diameter(index, 1);
        if (diameter <= threshold) edges.push_back({diameter, index});
    }
    return edges;
}

void ripser::gpu_compute_dim_0_pairs(std::vector<struct diameter_index_t_struct>& columns_to_reduce) {
    union_find dset(n);

    index_t max_num_edges= binomial_coeff(n, 2);
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare_reverse cmp_reverse;
    cudaMemset(d_flagarray_OR_index_to_subindex, 0, sizeof(index_t)*max_num_edges);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_edges<index_t>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_edges<<<grid_size, 256>>>(d_flagarray_OR_index_to_subindex, d_columns_to_reduce, threshold, d_distance_matrix, max_num_edges, n, d_binomial_coeff);
    CUDACHECK(cudaDeviceSynchronize());

    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex+max_num_edges, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_edges, cmp_reverse);


    cudaMemcpy(h_columns_to_reduce, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);

    std::cout << "persistence intervals in dim 0:" << std::endl;

    std::vector<index_t> vertices_of_edge(2);
    for(index_t idx=0; idx<*h_num_columns_to_reduce; idx++){
        struct diameter_index_t_struct e= h_columns_to_reduce[idx];
        vertices_of_edge.clear();
        get_simplex_vertices(e.index, 1, n, std::back_inserter(vertices_of_edge));
        index_t u= dset.find(vertices_of_edge[0]), v= dset.find(vertices_of_edge[1]);

        if (u != v) {
            //remove paired destroyer columns (we compute cohomology)
            if(e.diameter!=0) {
                std::cout << " [0," << e.diameter << ")" << std::endl;
                birth_death_coordinate barcode = {0,e.diameter};
                list_of_barcodes[0].push_back(barcode);
            }
            dset.link(u, v);
        } else {
            columns_to_reduce.push_back(e);
        }
    }
    std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());

    //don't want to reverse the h_columns_to_reduce so just put into vector and copy later
    #pragma omp parallel for schedule(guided,1)
    for(index_t i=0; i<columns_to_reduce.size(); i++){
        h_columns_to_reduce[i]= columns_to_reduce[i];
    }
    *h_num_columns_to_reduce= columns_to_reduce.size();
    *h_num_nonapparent= *h_num_columns_to_reduce;//we haven't found any apparent columns yet, so set all columns to nonapparent

    for(index_t i= 0; i < n; ++i) {
        if(dset.find(i) == i) std::cout << " [0, )" << std::endl << std::flush;
    }
}

//finding apparent pairs
void ripser::gpuscan(const index_t dim) {
    //(need to sort for filtration order before gpuscan first, then apply gpu scan then sort again)
    //note: scan kernel can eliminate high percentage of columns in little time.
    //filter by fully reduced columns (apparent pairs) found by gpu scan

    //need this to prevent 0-blocks kernels from executing
    if(*h_num_columns_to_reduce==0) {
        return;
    }
    index_t num_simplices= binomial_coeff(n, dim+1);
    cudaMemcpy(d_columns_to_reduce, h_columns_to_reduce, sizeof(struct diameter_index_t_struct) * *h_num_columns_to_reduce, cudaMemcpyHostToDevice);

    CUDACHECK(cudaDeviceSynchronize());

    //@nitish
    thrust::fill(thrust::device, d_cidx_to_diameter, d_cidx_to_diameter + num_simplices, -MAX_FLOAT);
    CUDACHECK(cudaDeviceSynchronize());

    //@nitish
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, init_cidx_to_diam, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    //there will be kernel launch errors if columns_to_reduce.size()==0; it causes thrust to complain later in the code execution

    //@nitish
    init_cidx_to_diam<<<grid_size, 256>>>(d_cidx_to_diameter, d_columns_to_reduce, *h_num_columns_to_reduce);

    CUDACHECK(cudaDeviceSynchronize());

    cudaMemset(d_lowest_one_of_apparent_pair, -1, sizeof(index_t) * *h_num_columns_to_reduce);
    CUDACHECK(cudaDeviceSynchronize());

    //@nitish
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, coboundary_findapparent_single_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    coboundary_findapparent_single_kernel<<<grid_size, 256, 256 * (dim + 1) * sizeof(index_t)>>>(d_cidx_to_diameter, d_columns_to_reduce, d_lowest_one_of_apparent_pair, dim, num_simplices, n, d_binomial_coeff, *h_num_columns_to_reduce, d_distance_matrix, threshold);

    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaDeviceSynchronize());

    //post processing (inserting appararent pairs into a "hash map": 2 level data structure) now on GPU
    struct row_cidx_column_idx_struct_compare cmp_pivots;

    //put pairs into an array

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, gpu_insert_pivots_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    gpu_insert_pivots_kernel<<<grid_size, 256>>>(d_pivot_array, d_lowest_one_of_apparent_pair, d_pivot_column_index_OR_nonapparent_cols, *h_num_columns_to_reduce, d_num_nonapparent);
    CUDACHECK(cudaDeviceSynchronize());

    thrust::sort(thrust::device, d_pivot_array, d_pivot_array+*h_num_columns_to_reduce, cmp_pivots);
    thrust::sort(thrust::device, d_pivot_column_index_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols+*h_num_nonapparent);

    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
    //transfer to CPU side all GPU data structures
    cudaMemcpy(h_pivot_array, d_pivot_array, sizeof(index_t_pair_struct)*(num_apparent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivot_column_index_array_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols, sizeof(index_t)*(*h_num_nonapparent), cudaMemcpyDeviceToHost);

    cudaMemset(d_flagarray_OR_index_to_subindex, -1, sizeof(index_t)* *h_num_columns_to_reduce);
    //perform the scatter operation
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, init_index_to_subindex, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    init_index_to_subindex<<<grid_size, 256>>>(d_flagarray_OR_index_to_subindex, d_pivot_column_index_OR_nonapparent_cols, *h_num_nonapparent);
    cudaMemcpy(h_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex, sizeof(index_t)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);
}

//finding apparent pairs
void ripser::gpu_assemble_columns_to_reduce_plusplus(const index_t dim) {

    index_t max_num_simplices= binomial_coeff(n, dim + 1);

    Stopwatch sw;
    sw.start();

#pragma omp parallel for schedule(guided,1)
    for (index_t i= 0; i < max_num_simplices; i++) {
#ifdef USE_PHASHMAP
        h_pivot_column_index_array_OR_nonapparent_cols[i]= phmap_get_value(i);
#endif
#ifdef USE_GOOGLE_HASHMAP
        auto pair= pivot_column_index.find(i);
        if(pair!=pivot_column_index.end()){
            h_pivot_column_index_array_OR_nonapparent_cols[i]= pair->second;
        }else{
            h_pivot_column_index_array_OR_nonapparent_cols[i]= -1;
        }
#endif
    }
    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
    if(num_apparent>0) {
#pragma omp parallel for schedule(guided, 1)
        for (index_t i= 0; i < num_apparent; i++) {
            index_t row_cidx= h_pivot_array[i].row_cidx;
            h_pivot_column_index_array_OR_nonapparent_cols[row_cidx]= h_pivot_array[i].column_idx;
        }
    }
    *h_num_columns_to_reduce= 0;
    cudaMemcpy(d_pivot_column_index_OR_nonapparent_cols, h_pivot_column_index_array_OR_nonapparent_cols, sizeof(index_t)*max_num_simplices, cudaMemcpyHostToDevice);

    sw.stop();
#ifdef PROFILING
    std::cerr<<"time to copy hash map for dim "<<dim<<": "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    cudaMemset(d_flagarray_OR_index_to_subindex, 0, sizeof(index_t)*max_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());
#else
    cudaMemset(d_flagarray, 0, sizeof(char)*max_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());
#endif
    Stopwatch pop_cols_timer;
    pop_cols_timer.start();

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_columns_to_reduce<index_t>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_columns_to_reduce<<<grid_size, 256, 256 * (dim + 1) * sizeof(index_t)>>>(d_flagarray_OR_index_to_subindex, d_columns_to_reduce, d_pivot_column_index_OR_nonapparent_cols, d_distance_matrix, n, max_num_simplices, dim, threshold, d_binomial_coeff);
#else
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_columns_to_reduce<char>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_columns_to_reduce<<<grid_size, 256, 256 * (dim + 1) * sizeof(index_t)>>>(d_flagarray, d_columns_to_reduce, d_pivot_column_index_OR_nonapparent_cols, d_distance_matrix, n, max_num_simplices, dim, threshold, d_binomial_coeff);
#endif
    CUDACHECK(cudaDeviceSynchronize());
    pop_cols_timer.stop();

    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex+max_num_simplices, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_simplices, cmp);
#else
    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray, d_flagarray+max_num_simplices, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_simplices, cmp);
#endif

#ifdef COUNTING
    std::cerr<<"num cols to reduce for dim "<<dim<<": "<<*h_num_columns_to_reduce<<std::endl;
#endif
    cudaMemcpy(h_columns_to_reduce, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);

}

void ripser::cpu_byneighbor_assemble_columns_to_reduce(std::vector<diameter_index_t_struct>& simplices, std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t,index_t>& pivot_column_index, index_t dim) {
    --dim;
    columns_to_reduce.clear();
    std::vector<struct diameter_index_t_struct> next_simplices;

    for(struct diameter_index_t_struct& simplex : simplices) {
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while(cofacets.has_next(false)) {
            auto cofacet= cofacets.next();
            if(cofacet.diameter <= threshold) {
                next_simplices.push_back(cofacet);
                if(pivot_column_index.find(cofacet.index) == pivot_column_index.end()) {
                    columns_to_reduce.push_back(cofacet);
                }
            }
        }
    }

    simplices.swap(next_simplices);
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(), cmp);
}

void ripser::assemble_columns_gpu_accel_transition_to_cpu_only(const bool& more_than_one_dim_cpu_only,std::vector<diameter_index_t_struct>& simplices, std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t,index_t>& cpu_pivot_column_index, index_t dim){
    index_t max_num_simplices= binomial_coeff(n,dim+1);
    //insert all pivots from the two gpu pivot data structures into cpu_pivot_column_index, cannot parallelize this for loop due to concurrency issues of hashmaps
    for(index_t i= 0; i < max_num_simplices; i++) {
        index_t col_idx= phmap_get_value(i);
        if(col_idx!=-1) {
            cpu_pivot_column_index[i]= col_idx;
        }
    }

    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
    if(num_apparent>0) {
        //we can't insert into the hashmap in parallel
        for(index_t i= 0; i < num_apparent; i++) {
            index_t row_cidx= h_pivot_array[i].row_cidx;
            index_t column_idx= h_pivot_array[i].column_idx;
            if(column_idx!=-1) {
                cpu_pivot_column_index[row_cidx]= column_idx;
            }
        }
    }

    columns_to_reduce.clear();
    simplices.clear();
    index_t count_simplices= 0;
    //cpu_pivot_column_index can't be parallelized for lookup
    for(index_t index = 0; index < max_num_simplices; ++index) {
        value_t diameter = -MAX_FLOAT;

        //the second condition after the || should never happen, since we never insert such pairs into cpu_pivot_column_index
        if(cpu_pivot_column_index.find(index) == cpu_pivot_column_index.end() || cpu_pivot_column_index[index]==-1) {
            diameter= compute_diameter(index, dim);
            if(diameter <= threshold) {
                columns_to_reduce.push_back({diameter, index});
            }
        }

        if(more_than_one_dim_cpu_only) {
            if(diameter==-MAX_FLOAT) {
                diameter= compute_diameter(index, dim);
            }
            if(diameter<=threshold) {
                simplices.push_back({diameter,index});
                count_simplices++;
            }
        }
    }

    greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(), cmp);
}

index_t ripser::get_value_pivot_array_hashmap(index_t row_cidx, struct row_cidx_column_idx_struct_compare cmp) {
#ifdef USE_PHASHMAP
    index_t col_idx= phmap_get_value(row_cidx);
    if(col_idx==-1){
#endif
#ifdef USE_GOOGLE_HASHMAP
        auto pair= pivot_column_index.find(row_cidx);
        if(pair==pivot_column_index.end()){
#endif
        index_t first= 0;
        index_t last= num_apparent- 1;

        while(first<=last){
            index_t mid= first + (last-first)/2;
            if(h_pivot_array[mid].row_cidx==row_cidx){
                return h_pivot_array[mid].column_idx;
            }
            if(h_pivot_array[mid].row_cidx<row_cidx){
                first= mid+1;
            }else{
                last= mid-1;
            }
        }
        return -1;

    }else{

#ifdef USE_PHASHMAP
        return col_idx;
#endif
#ifdef USE_GOOGLE_HASHMAP
        return pair->second;
#endif
    }
}

void ripser::compute_dim_0_pairs(std::vector<diameter_index_t_struct>& edges, std::vector<diameter_index_t_struct>& columns_to_reduce) {
#ifdef PRINT_PERSISTENCE_PAIRS
    std::cout << "persistence intervals in dim 0:" << std::endl;
#endif

    union_find dset(n);

    edges= get_edges();
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;

    std::sort(edges.rbegin(), edges.rend(), cmp);

    std::vector<index_t> vertices_of_edge(2);
    for (auto e : edges) {
        get_simplex_vertices(e.index, 1, n, vertices_of_edge.rbegin());
        index_t u= dset.find(vertices_of_edge[0]), v= dset.find(vertices_of_edge[1]);

        if (u != v) {
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
            if(e.diameter!=0) {
#ifdef PRINT_PERSISTENCE_PAIRS
                std::cout << " [0," << e.diameter << ")" << std::endl;
#endif
                //Collect persistence pair
                birth_death_coordinate barcode = {0,e.diameter};
                list_of_barcodes[0].push_back(barcode);
            }
#endif
            dset.link(u, v);
        } else {
            columns_to_reduce.push_back(e);
        }
    }
    std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());

#ifdef PRINT_PERSISTENCE_PAIRS
    for (index_t i= 0; i < n; ++i)
        if (dset.find(i) == i) std::cout << " [0, )" << std::endl;
#endif
}

void ripser::compute_barcodes() {
    index_t gpu_dim_max = calculate_gpu_dim_max_for_fullrips_computation_from_memory(dim_max, true);

    max_num_simplices_forall_dims= gpu_dim_max<(n/2)-1?get_num_simplices_for_dim(gpu_dim_max): get_num_simplices_for_dim((n/2)-1);
    if(1 <= gpu_dim_max) {

        CUDACHECK(cudaMalloc((void **) &d_columns_to_reduce, sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims));
        h_columns_to_reduce= (struct diameter_index_t_struct*) malloc(sizeof(struct diameter_index_t_struct)* max_num_simplices_forall_dims);

        if(h_columns_to_reduce==nullptr){
            std::cerr<<"malloc for h_columns_to_reduce failed"<<std::endl;
            exit(1);
        }

        CUDACHECK(cudaMalloc((void **) &d_cidx_to_diameter, sizeof(value_t)*max_num_simplices_forall_dims));
        CUDACHECK(cudaMalloc((void **) &d_flagarray_OR_index_to_subindex, sizeof(index_t)*max_num_simplices_forall_dims));

        h_flagarray_OR_index_to_subindex = (index_t*) malloc(sizeof(index_t)*max_num_simplices_forall_dims);
        if(h_flagarray_OR_index_to_subindex == nullptr) {
            std::cerr<<"malloc for h_index_to_subindex failed"<<std::endl;
        }
        CUDACHECK(cudaMalloc((void **) &d_distance_matrix, sizeof(value_t)*dist.size()*(dist.size()-1)/2));
        cudaMemcpy(d_distance_matrix, dist.distances.data(), sizeof(value_t)*dist.size()*(dist.size()-1)/2, cudaMemcpyHostToDevice);

        CUDACHECK(cudaMalloc((void **) &d_pivot_column_index_OR_nonapparent_cols, sizeof(index_t)*max_num_simplices_forall_dims));

        //this array is used for both the pivot column index hash table array as well as the nonapparent cols array as an unstructured hashmap
        h_pivot_column_index_array_OR_nonapparent_cols= (index_t*) malloc(sizeof(index_t)*max_num_simplices_forall_dims);

        if(h_pivot_column_index_array_OR_nonapparent_cols == nullptr) {
            std::cerr<<"malloc for h_pivot_column_index_array_OR_nonapparent_cols failed"<<std::endl;
            exit(1);
        }

        //copy object over to GPU
        CUDACHECK(cudaMalloc((void**) &d_binomial_coeff, sizeof(binomial_coeff_table)));
        cudaMemcpy(d_binomial_coeff, &binomial_coeff, sizeof(binomial_coeff_table), cudaMemcpyHostToDevice);

        index_t num_binoms= binomial_coeff.get_num_n()*binomial_coeff.get_max_tuple_length();

        CUDACHECK(cudaMalloc((void **) &h_d_binoms, sizeof(index_t)*num_binoms));
        cudaMemcpy(h_d_binoms, binomial_coeff.binoms, sizeof(index_t)*num_binoms, cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_binomial_coeff->binoms), &h_d_binoms, sizeof(index_t*), cudaMemcpyHostToDevice);

        cudaHostAlloc((void **)&h_num_columns_to_reduce, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_columns_to_reduce, h_num_columns_to_reduce,0);
        cudaHostAlloc((void **)&h_num_nonapparent, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_nonapparent, h_num_nonapparent,0);

        CUDACHECK(cudaMalloc((void**) &d_lowest_one_of_apparent_pair, sizeof(index_t)*max_num_simplices_forall_dims));
        CUDACHECK(cudaMalloc((void**) &d_pivot_array, sizeof(struct index_t_pair_struct)*max_num_simplices_forall_dims));
        h_pivot_array = (struct index_t_pair_struct*) malloc(sizeof(struct index_t_pair_struct)*max_num_simplices_forall_dims);
        if(h_pivot_array == nullptr) {
            std::cerr<<"malloc for h_pivot_array failed"<<std::endl;
            exit(1);
        }
    }

    columns_to_reduce.clear();
    std::vector<diameter_index_t_struct> simplices;
    if(1 <= gpu_dim_max) {
        gpu_compute_dim_0_pairs(columns_to_reduce);
    }
    else {
        compute_dim_0_pairs(simplices, columns_to_reduce);
    }

    index_t dim_forgpuscan = 1;
    for(index_t dim = 1; dim <= gpu_dim_max; ++dim) {
        phmap_clear();

        *h_num_nonapparent= 0;
        //search for apparent pairs
        gpuscan(dim);
        //dim_forgpuscan= dim;//update dim_forgpuscan to the dimension that gpuscan was just done at
        compute_pairs_plusplus(dim, dim_forgpuscan);
        if(dim < gpu_dim_max) {
            gpu_assemble_columns_to_reduce_plusplus(dim+1);
        }
    }

    if(gpu_dim_max < dim_max) {//do cpu only computation from this point on
        std::cerr<<"CPU-ONLY MODE FOR REMAINDER OF HIGH DIMENSIONAL COMPUTATION (NOT ENOUGH GPU DEVICE MEMORY)"<<std::endl;
        free_init_cpumem();
        hash_map<index_t,index_t> cpu_pivot_column_index;
        cpu_pivot_column_index.reserve(*h_num_columns_to_reduce);
        bool more_than_one_dim_to_compute = dim_max>gpu_dim_max+1;
        assemble_columns_gpu_accel_transition_to_cpu_only(more_than_one_dim_to_compute, simplices, columns_to_reduce, cpu_pivot_column_index, gpu_dim_max+1);
        free_remaining_cpumem();
        for(index_t dim = gpu_dim_max+1; dim <= dim_max; ++dim) {
            cpu_pivot_column_index.clear();
            cpu_pivot_column_index.reserve(columns_to_reduce.size());
            compute_pairs(columns_to_reduce, cpu_pivot_column_index, dim);
            if(dim<dim_max) {
                cpu_byneighbor_assemble_columns_to_reduce(simplices, columns_to_reduce, cpu_pivot_column_index, dim+1);
            }
        }
    }
    else {
        if (n >= 10) {
            free_init_cpumem();
            free_remaining_cpumem();
        }
    }

    if(gpu_dim_max>=1 && n>=10) {
        free_gpumem_dense_computation();
        cudaFreeHost(h_num_columns_to_reduce);
        cudaFreeHost(h_num_nonapparent);
        free(h_flagarray_OR_index_to_subindex);
    }
}

template <typename T> T read(std::istream& s) {
    T result;
    s.read(reinterpret_cast<char*>(&result), sizeof(T));
    return result; // on little endian: boost::endian::little_to_native(result);
}

compressed_lower_distance_matrix read_point_cloud(std::istream& input_stream) {
    std::vector<std::vector<value_t>> points;

    std::string line;
    value_t value;
    while (std::getline(input_stream, line)) {
        std::vector<value_t> point;
        std::istringstream s(line);
        while (s >> value) {
            point.push_back(value);
            s.ignore();
        }
        if (!point.empty()) points.push_back(point);
        assert(point.size() == points.front().size());
    }

    euclidean_distance_matrix eucl_dist(std::move(points));

    index_t n= eucl_dist.size();
#ifdef COUNTING
    std::cout << "point cloud with " << n << " points in dimension "
              << eucl_dist.points.front().size() << std::endl;
#endif
    std::vector<value_t> distances;

    for (int i= 0; i < n; ++i)
        for (int j= 0; j < i; ++j) distances.push_back(eucl_dist(i, j));

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_file(std::istream& input_stream) {
    return read_point_cloud(input_stream);
}

void print_usage_and_exit(int exit_code) {
    std::cerr
            << "Usage: "
            << "ripser++ "
            << "[options] [filename]" << std::endl
            << std::endl
            << "Options:" << std::endl
            << std::endl
            << "  --help           print this screen" << std::endl
            << "  --format         use the specified file format for the input. Options are:"
            << std::endl
            << "                     lower-distance (lower triangular distance matrix; default)"
            << std::endl
            << "                     distance       (full distance matrix)" << std::endl
            << "                     point-cloud    (point cloud in Euclidean space)" << std::endl
            << "                     dipha          (distance matrix in DIPHA file format)" << std::endl
            << "                     sparse         (sparse distance matrix in sparse triplet (COO) format)"
            << std::endl
            << "                     binary         (distance matrix in Ripser binary file format)"
            << std::endl
            << "  --dim <k>        compute persistent homology up to dimension <k>" << std::endl
            << "  --threshold <t>  compute Rips complexes up to diameter <t>" << std::endl
            << "  --sparse         force sparse computation "<<std::endl
            << "  --ratio <r>      only show persistence pairs with death/birth ratio > r" << std::endl
            << std::endl;

    exit(exit_code);
}

#ifndef HH_LIB
int main(int argc, char** argv) {
    const char* filename = nullptr;
    index_t dim_max      = 1;
    value_t threshold    = std::numeric_limits<value_t>::max();
    float ratio          = 1;
    bool use_sparse      = false;
    for(index_t i= 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if(arg == "--help") {
            print_usage_and_exit(0);
        }
        else if (arg == "--dim") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            dim_max= std::stol(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        }
        else if (arg == "--threshold") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            threshold= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        }
        else if (arg == "--ratio") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            ratio= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        }
        else if(arg=="--sparse") {
            use_sparse= true;
        }
        else {
            if (filename) { print_usage_and_exit(-1); }
            filename= argv[i];
        }
    }

    std::ifstream file_stream(filename);
    if (filename && file_stream.fail()) {
        std::cerr << "couldn't open file " << filename << std::endl;
        exit(-1);
    }

    compressed_lower_distance_matrix dist = read_file(filename ? file_stream : std::cin);

    value_t enclosing_radius = std::numeric_limits<value_t>::infinity();
    for(index_t i = 0; i < dist.size(); ++i) {
        value_t r_i = -std::numeric_limits<value_t>::infinity();
        for(index_t j = 0; j < dist.size(); ++j) r_i = std::max(r_i, dist(i, j));
        enclosing_radius = std::min(enclosing_radius, r_i);
    }

    if(threshold == std::numeric_limits<value_t>::max()) threshold = enclosing_radius;

    ripser(std::move(dist), dim_max, threshold, ratio).compute_barcodes();
}
#endif
