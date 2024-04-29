#include <hedgehog/hedgehog.h>
#include "hh.h"
#include <limits>

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

    auto ripserpp = ripser(std::move(dist), dim_max, threshold, ratio);
    ripserpp.compute_barcodes();
    //cudaDeviceReset();

    return 0;
}
