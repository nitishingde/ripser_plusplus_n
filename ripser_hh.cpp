#include <atomic>
#include <hedgehog/hedgehog.h>
#include "hh.h"
#include <limits>
#include <phmap_interface/phmap_interface.h>

struct GlobalContext {
    ripser               *pRipser      = nullptr;
    index_t              gpu_dim_max   = -1;
    index_t              currentDim    = 0;
    index_t              num_simplices = -1;
    cudaEvent_t          gpuScanEvent[4];

    explicit GlobalContext() {
        for(auto &cudaEvent: gpuScanEvent) {
            cudaEventCreate(&cudaEvent);
        }
    }

    ~GlobalContext() {
        for(auto cudaEvent: gpuScanEvent) {
            cudaEventDestroy(cudaEvent);
        }
    }
};

class InitTask: public hh::AbstractTask<1, GlobalContext, GlobalContext> {
public:
    explicit InitTask(): hh::AbstractTask<1, GlobalContext, GlobalContext>("Init Task", 1, false) {}
    void execute(std::shared_ptr<GlobalContext> ctx) override {
        auto pRipser = ctx->pRipser;
        auto &gpu_dim_max = ctx->gpu_dim_max;
        ctx->currentDim = 0;
        gpu_dim_max = pRipser->calculate_gpu_dim_max_for_fullrips_computation_from_memory(pRipser->dim_max, true);

        std::vector<struct diameter_index_t_struct> columns_to_reduce;
        pRipser->init(gpu_dim_max);

        columns_to_reduce.clear();
        std::vector<diameter_index_t_struct> simplices;
        if(1 <= gpu_dim_max) {
            pRipser->gpu_compute_dim_0_pairs(columns_to_reduce);
        }
        else {
            pRipser->compute_dim_0_pairs(simplices, columns_to_reduce);
        }
        this->addResult(ctx);
    }
};

class StartLoopTask: public hh::AbstractTask<1, GlobalContext, GlobalContext> {
public:
    explicit StartLoopTask(): hh::AbstractTask<1, GlobalContext, GlobalContext>("Loop Task", 1, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        auto pRipser = ctx->pRipser;
        auto gpu_dim_max = ctx->gpu_dim_max;
        ctx->currentDim++;
        if(ctx->currentDim <= ctx->gpu_dim_max) {
            phmap_clear();
            pRipser->set_h_num_nonapparent(0);
            ctx->num_simplices = pRipser->binomial_coeff(pRipser->n, ctx->currentDim+1);
            this->addResult(ctx);
            return;
        }

        if (10 <= pRipser->n) {
            pRipser->free_init_cpumem();
            pRipser->free_remaining_cpumem();
        }

        if(1 <= gpu_dim_max && 10 <= pRipser->n) {
            pRipser->free_gpumem_dense_computation();
            cudaFreeHost(pRipser->h_num_columns_to_reduce);
            cudaFreeHost(pRipser->h_num_nonapparent);
//            free(pRipser->h_flagarray_OR_index_to_subindex);
        }
        canTerminate_.store(true);
    }

    [[nodiscard]] bool canTerminate() const override {
        return canTerminate_.load();
    }

private:
    std::atomic<bool> canTerminate_ = false;
};

class GpuScanTask0: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanTask0(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("GpuScan Task0", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        auto pRipser = ctx->pRipser;
        pRipser->gpuscan_0(ctx->currentDim, ctx->num_simplices, this->stream());
        cudaEventRecord(ctx->gpuScanEvent[0], this->stream());
        this->addResult(ctx);
    }
};

class GpuScanTask1: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanTask1(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("GpuScan Task1", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        auto pRipser = ctx->pRipser;
        pRipser->gpuscan_1(ctx->currentDim, ctx->num_simplices, this->stream());
        cudaEventRecord(ctx->gpuScanEvent[1], this->stream());
        this->addResult(ctx);
    }
};

class GpuScanTask2: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanTask2(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("GpuScan Task2", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        auto pRipser = ctx->pRipser;
        pRipser->gpuscan_2(ctx->currentDim, ctx->num_simplices, this->stream());
        cudaEventRecord(ctx->gpuScanEvent[2], this->stream());
        this->addResult(ctx);
    }
};

class GpuScanTask3: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanTask3(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("GpuScan Task3", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        ttl++;
        if(ttl%2 != 0) return;

        auto pRipser = ctx->pRipser;
        cudaEventSynchronize(ctx->gpuScanEvent[0]);
        cudaEventSynchronize(ctx->gpuScanEvent[1]);
        pRipser->gpuscan_3(ctx->currentDim, ctx->num_simplices, this->stream());
        cudaEventRecord(ctx->gpuScanEvent[3], this->stream());
        this->addResult(ctx);
    }
private:
    int32_t ttl = 0;
};

class GpuScanTask4: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanTask4(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("GpuScan 4", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        ttl++;
        if(ttl%2 != 0) return;

        auto pRipser = ctx->pRipser;
        cudaEventSynchronize(ctx->gpuScanEvent[2]);
        cudaEventSynchronize(ctx->gpuScanEvent[3]);
        pRipser->gpuscan_4(ctx->currentDim, ctx->num_simplices, this->stream());
        cudaStreamSynchronize(this->stream());
        this->addResult(ctx);
    }
private:
    int32_t ttl = 0;
};

class GpuScanGraph: public hh::Graph<1, GlobalContext, GlobalContext> {
public:
    explicit GpuScanGraph(): hh::Graph<1, GlobalContext, GlobalContext>("GpuScan Graph") {
        auto task0 = std::make_shared<GpuScanTask0>();
        auto task1 = std::make_shared<GpuScanTask1>();
        auto task2 = std::make_shared<GpuScanTask2>();
        auto task3 = std::make_shared<GpuScanTask3>();
        auto task4 = std::make_shared<GpuScanTask4>();

        this->input<GlobalContext>(task0);
        this->input<GlobalContext>(task1);
        this->input<GlobalContext>(task2);
        this->edge<GlobalContext>(task0, task3);
        this->edge<GlobalContext>(task1, task3);
        this->edge<GlobalContext>(task2, task4);
        this->edge<GlobalContext>(task3, task4);
        this->output<GlobalContext>(task4);
    }
};

class ComputePairsTask: public hh::AbstractTask<1, GlobalContext, GlobalContext> {
public:
    explicit ComputePairsTask(): hh::AbstractTask<1, GlobalContext, GlobalContext>("ComputePairs Task", 1, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        ctx->pRipser->compute_pairs_plusplus(ctx->currentDim, 1);
        this->addResult(ctx);
    }
};

class ReduceColumnTask: public hh::AbstractCUDATask<1, GlobalContext, GlobalContext> {
public:
    explicit ReduceColumnTask(): hh::AbstractCUDATask<1, GlobalContext, GlobalContext>("ReduceColumn Task", 1, false, false) {}

    void execute(std::shared_ptr<GlobalContext> ctx) override {
        if(ctx->currentDim < ctx->gpu_dim_max) {
            ctx->pRipser->gpu_assemble_columns_to_reduce_plusplus(ctx->currentDim+1, this->stream());
        }
        this->addResult(ctx);
    }
};

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

    auto ctx = std::make_shared<GlobalContext>();
    auto ripserpp = ripser(std::move(dist), dim_max, threshold, ratio);
    ctx->pRipser = &ripserpp;
    ctx->currentDim = 0;

    auto initTask = std::make_shared<InitTask>();
    auto loopTask = std::make_shared<StartLoopTask>();
    auto scanGraph = std::make_shared<GpuScanGraph>();
    auto computePairsTask = std::make_shared<ComputePairsTask>();
    auto reduceColumnTask = std::make_shared<ReduceColumnTask>();

    auto graph = hh::Graph<1, GlobalContext, GlobalContext>("Ripser + Hedgehog Dataflow graph");
    graph.input<GlobalContext>(initTask);
    graph.edge<GlobalContext>(initTask, loopTask);
    graph.edge<GlobalContext>(loopTask, scanGraph);
    graph.edge<GlobalContext>(scanGraph, computePairsTask);
    graph.edge<GlobalContext>(computePairsTask, reduceColumnTask);
    graph.edge<GlobalContext>(reduceColumnTask, loopTask);
    graph.output<GlobalContext>(loopTask);
    graph.executeGraph();
    graph.pushData(ctx);
    graph.finishPushingData();
    graph.waitForTermination();
    graph.createDotFile("ripser.dot", hh::ColorScheme::EXECUTION);

    return 0;
}
