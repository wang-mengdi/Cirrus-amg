#include "SimpleSolver.h"
#include "OperatorChecks.h"
#include "ProjectionSolver.h"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "Usage: simple_channel path/to/channel.json\n";
            return 2;
        }
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open case JSON");
        nlohmann::json config; input >> config;
        auto options = simple::Options::fromJson(config);
        if(config.contains("projection_replay")) {
            const auto& probe=config.at("projection_replay");
            if(config.contains("pressure_replay") || config.value("geometry_only",false) ||
               !config.value("operator_only",false) || options.fluidSolver!="proj" ||
               options.linearBackend!="native_gpu" || options.gpuPreconditioner!="native_amg" ||
               options.gpuPressureGauge!="mean_zero" || options.gpuPressureOperator!="full" ||
               !options.restartCheckpoint.empty() || !probe.is_object() ||
               (probe.contains("flux")==probe.contains("predictor_state")) ||
               (probe.contains("flux") && !probe.at("flux").is_string()) ||
               (probe.contains("predictor_state") && !probe.at("predictor_state").is_string()) ||
               (probe.contains("repetitions") && !probe.at("repetitions").is_number_integer()))
                throw std::runtime_error("Projection replay requires operator_only, full mean-zero native AMG projection, a flux path and no flow restart or other probe");
            const int count=probe.value("repetitions",1);
            if(count<1 || count>4)throw std::runtime_error("Projection replay repetitions must be between 1 and 4");
        }
        if(config.contains("pressure_replay")) {
            const auto& probe=config.at("pressure_replay");
            if(!config.value("operator_only",false) || options.fluidSolver!="proj" ||
               options.linearBackend!="native_gpu" || options.gpuPreconditioner!="native_amg" ||
               options.gpuPressureGauge!="mean_zero" || options.gpuPressureOperator!="full" ||
               !options.restartCheckpoint.empty() || !probe.is_object() ||
               !probe.contains("rhs") || !probe.at("rhs").is_string() ||
               (probe.contains("repetitions") && !probe.at("repetitions").is_number_integer()))
                throw std::runtime_error("Pressure replay requires operator_only, full mean-zero native AMG projection, a RHS path and no flow restart");
            const int count=probe.value("repetitions",1);
            if(count<1 || count>16)throw std::runtime_error("Pressure replay repetitions must be between 1 and 16");
        }
        std::filesystem::create_directories(options.output);
        std::ofstream(options.output / "case.json") << config.dump(2) << '\n';
        auto mesh = options.embeddedGeometry.empty() ?
            simple::makeOctreeChannel(options.ny, options.adaptive, options.periodicX, options.periodicZ) :
            simple::makeEmbeddedOctree(options.embeddedGeometry,options.adaptive);
        if(config.value("geometry_only",false)) {
            simple::dumpMeshCsv(mesh,(options.output/"mesh").string());
            std::cout<<"Geometry ready: "<<mesh.cells.size()<<" cells, "<<mesh.faces.size()<<" faces, "<<mesh.coarseFineFaces<<" coarse/fine faces\n";
            return 0;
        }
        if(!mesh.embedded) simple::runOperatorChecks(mesh, (options.output / "operator_checks.json").string());
        if(options.fluidSolver=="proj") {
            simple::ProjectionSolver solver(std::move(mesh),options);
            if(config.contains("projection_replay")) {
                const auto& probe=config.at("projection_replay");
                const auto report=probe.contains("predictor_state")?
                    solver.replayPredictor(probe.at("predictor_state").get<std::string>(),probe.value("repetitions",1)):
                    solver.replayProjection(probe.at("flux").get<std::string>(),probe.value("repetitions",1));
                return report.at("all_projection_checks_passed").get<bool>()?0:3;
            }
            if(config.contains("pressure_replay")) {
                const auto& probe=config.at("pressure_replay");
                return solver.replayPressure(probe.at("rhs").get<std::string>(),probe.value("repetitions",1))
                    .at("all_solve_checks_passed").get<bool>()?0:3;
            }
            if(config.value("operator_only",false)) {
                std::cout<<"Projection operators initialized; no flow iteration was run\n";
                return 0;
            }
            return solver.run().at("converged").get<bool>()?0:3;
        }
        simple::Solver solver(std::move(mesh), options);
        if(config.value("operator_only",false)) {
            std::cout<<"Operator checks ready; no flow iteration was run\n";
            return 0;
        }
        auto report = solver.run();
        return report.at("converged").get<bool>() ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "SIMPLE error: " << e.what() << '\n'; return 1;
    } catch (const std::string& e) {
        std::cerr << "SIMPLE error: " << e << '\n'; return 1;
    }
}
