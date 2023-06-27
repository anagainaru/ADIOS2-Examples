#include <ios>
#include <iostream>
#include <vector>
#include <numeric>

#include <adios2.h>

#include <Kokkos_Core.hpp>
#include <adios2/cxx11/KokkosView.h>

int BPWrite(const std::string fname, const size_t N, int nSteps,
            const std::string engine)
{
    // Initialize the simulation data with the default memory space
    using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<float *, mem_space> gpuSimData("simBuffer", N);
    Kokkos::parallel_for(
        "initBuffer", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
        KOKKOS_LAMBDA(int i) { gpuSimData(i) = i; });
    Kokkos::fence();

    // Set up the ADIOS structures
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("WriteIO");
    io.SetEngine(engine);

    const adios2::Dims shape{static_cast<size_t>(N)};
    const adios2::Dims start{static_cast<size_t>(0)};
    const adios2::Dims count{N};
    auto data = io.DefineVariable<float>("data", shape, start, count);

    adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);


    adios2::Operator mgardOp =
        adios.DefineOperator("mgardCompressor", adios2::ops::LossyMGARD);
    data.AddOperation(mgardOp,
                         {{adios2::ops::mgard::key::tolerance, "0.0001"}});
    // Simulation steps
    for (size_t step = 0; step < nSteps; ++step)
    {
        // Make a 1D selection to describe the local dimensions of the
        // variable we write and its offsets in the global spaces
        adios2::Box<adios2::Dims> sel({0}, {N});
        data.SetSelection(sel);

        // Start IO step every write step
        bpWriter.BeginStep();
    std::chrono::steady_clock::time_point _start;
    _start = std::chrono::steady_clock::now();
        bpWriter.Put(data, gpuSimData.data());
        const auto duration = std::chrono::steady_clock::now() - _start;
        std::chrono::duration<double, std::milli> duration_ms(duration);
        std::cout << "Put on Device buffers with MGARD " << duration_ms.count() << std::endl;
        bpWriter.EndStep();

        // Update values in the simulation data using the default
        // execution space
        Kokkos::parallel_for(
            "updateBuffer",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
            KOKKOS_LAMBDA(int i) { gpuSimData(i) += 10; });
        Kokkos::fence();
    }

    bpWriter.Close();
    Kokkos::DefaultExecutionSpace exe_space;
    std::cout << "Done writing on memory space: " << exe_space.name()
              << std::endl;
    return 0;
}

int BPWriteCPU(const std::string fname, const size_t N, int nSteps,
            const std::string engine)
{
    // Initialize the simulation data with the default memory space
	std::vector<float> cpuSimData(N);
	std::iota(cpuSimData.begin(), cpuSimData.end(), 0);

    // Set up the ADIOS structures
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("WriteIO");
    io.SetEngine(engine);

    const adios2::Dims shape{static_cast<size_t>(N)};
    const adios2::Dims start{static_cast<size_t>(0)};
    const adios2::Dims count{N};
    auto data = io.DefineVariable<float>("data", shape, start, count);

    adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);


    adios2::Operator mgardOp =
        adios.DefineOperator("mgardCompressor", adios2::ops::LossyMGARD);
    data.AddOperation(mgardOp,
                         {{adios2::ops::mgard::key::tolerance, "0.0001"}});
    // Simulation steps
    for (size_t step = 0; step < nSteps; ++step)
    {
        // Make a 1D selection to describe the local dimensions of the
        // variable we write and its offsets in the global spaces
        adios2::Box<adios2::Dims> sel({0}, {N});
        data.SetSelection(sel);

        // Start IO step every write step
        bpWriter.BeginStep();
    std::chrono::steady_clock::time_point _start;
    _start = std::chrono::steady_clock::now();
        bpWriter.Put(data, cpuSimData.data());
        const auto duration = std::chrono::steady_clock::now() - _start;
        std::chrono::duration<double, std::milli> duration_ms(duration);
        std::cout << "Put on Host buffers with MGARD " << duration_ms.count() << std::endl;
        bpWriter.EndStep();

        // Update values in the simulation data using the default
	std::for_each(cpuSimData.begin(), cpuSimData.end(), [](float& d) { d+=10;});
    }

    bpWriter.Close();
    std::cout << "Done writing on memory space: Host"
              << std::endl;
    return 0;
}

int BPRead(const std::string fname, const size_t N, int nSteps,
           const std::string engine)
{
    // Create ADIOS structures
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("ReadIO");
    io.SetEngine(engine);

    Kokkos::DefaultExecutionSpace exe_space;
    std::cout << "Read on memory space: " << exe_space.name() << std::endl;

    adios2::Engine bpReader = io.Open(fname, adios2::Mode::Read);

    unsigned int step = 0;
    using mem_space = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::View<float *, mem_space> gpuSimData("simBuffer", N);
    for (; bpReader.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        auto data = io.InquireVariable<float>("data");
        const adios2::Dims start{0};
        const adios2::Dims count{N};
        const adios2::Box<adios2::Dims> sel(start, count);
        data.SetSelection(sel);

        bpReader.Get(data, gpuSimData.data());
        bpReader.EndStep();

        auto cpuData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           gpuSimData);
        std::cout << "Simualation step " << step << " : ";
        std::cout << cpuData.size() << " elements: " << cpuData[0];
        std::cout << " " << cpuData[1] << " ... ";
        std::cout << cpuData[cpuData.size() - 1] << std::endl;
    }

    bpReader.Close();
    return 0;
}

int main(int argc, char **argv)
{
    const std::vector<std::string> list_of_engines = {"BP5"};
    size_t N = 6000;
    if (argv[1])
        N = atoi(argv[1]);
    int nSteps = 10, ret = 0;

    Kokkos::initialize(argc, argv);
    {
        for (auto engine : list_of_engines)
        {
            std::cout << "Using engine " << engine << std::endl;
            const std::string fname(engine + "_Kokkos_WR.bp");
            ret += BPWrite(fname, N, nSteps, engine);
        }
    }
for (auto engine : list_of_engines)
{
    std::cout << "Using engine " << engine << std::endl;
    const std::string fname(engine + "_CPU_WR.bp");
    ret += BPWriteCPU(fname, N, nSteps, engine);
}
    Kokkos::finalize();
    return ret;
}
