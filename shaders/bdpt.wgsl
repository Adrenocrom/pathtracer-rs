// Moved: the BDPT compute shader now lives in the fragments in
// shaders/bdpt/ — common, intersect, spectral, sampling, bsdf,
// bridges, pathview, mis, lights, firefly, trace, main — concatenated
// in dependency order by BDPT_WGSL in src/gpu.rs (WGSL has no
// #include). This stub is kept only as a pointer for anything still
// looking for the old path; it contains no shader code and can be
// deleted outright.
