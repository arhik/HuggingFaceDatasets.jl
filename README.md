# HuggingFaceDatasets

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGenAI.github.io/HuggingFaceDatasets.jl/dev)
[![Build Status](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGenAI/HuggingFaceDatasets.jl) 

HuggingFaceDatasets.jl is a non-official julia wrapper around the python package  `datasets` from Hugging Face. `datasets` contains a large collection of machine learning datasets (see [here](https://huggingface.co/datasets) for a list) that this package makes available to the julia ecosystem.

This package is built on top of [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl).

## Installation

HuggingFaceDatasets.jl is a registered Julia package. You can easily install it through the package manager:

```julia
pkg> add HuggingFaceDatasets
```

## Usage

HuggingFaceDatasets.jl provides wrappers around types from the `datasets` python package (e.g. `Dataset` and `DatasetDict`) along with a few related methods.

Check out the [examples/](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl/tree/main/examples) folder for usage examples.

```julia
julia> using HuggingFaceDatasets

julia> train_data = load_dataset("mnist", split = "train")
Dataset({
    features: ['image', 'label'],
    num_rows: 60000
})

julia> train_data[1]
Python: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x3340B0290>, 'label': 5}

julia> length(train_data)
60000

julia> train_data = load_dataset("mnist", split = "train").with_format("julia");

```julia
julia> train_data[1] # Returned observations are now julia objects
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => Gray{N0f8}[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]

julia> train_data[1:2]
Dict{String, Vector} with 2 entries:
  "label" => [5, 0]
  "image" => ReinterpretArray{Gray{N0f8}, 2, UInt8, Matrix{UInt8}, false}[[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0…
```

## Streaming Datasets

For large datasets that don't fit in memory, use `streaming=true`:

```julia
julia> imdb = load_dataset("imdb", split="train", streaming=true)
IterableDataset({
    features: ['text', 'label'],
    num_shards: 1
})

julia> # Convert to Julia format and process
julia> imdb_julia = imdb.with_format("julia")

julia> # Stream processing methods
julia> first_sample = first(imdb_julia)
julia> first_five = collect(take(imdb_julia, 5))
julia> positive = Base.filter(sample -> sample["label"] == 1, imdb_julia)
julia> batches = batch(imdb_julia, 32)

julia> # ML training pipeline
julia> training = imdb_julia |>
           Base.filter(sample -> sample["label"] == 1) |>
           take(1000) |>
           batch(16)

for batch in training
    # Process 16 samples at a time
    println("Batch size: ", length(batch["label"]))
    break
end
```

### Streaming Methods

- `take(n)` - Get first `n` items
- `skip(n)` - Skip first `n` items  
- `shuffle(seed; buffer_size=1000)` - Shuffle with buffer
- `batch(batch_size)` - Create batches for ML
- `shard(num_shards, index)` - Split into parallel shards
- `filter(func)` - Filter items based on condition

| Feature | `Dataset` | `IterableDataset` (Streaming) |
|---------|-----------|------------------------------|
| Random Access | `ds[i]` ✅ | `ds[i]` ❌ |
| Length | `length(ds)` ✅ | Not available |
| Memory | Loads all data | Loads on-demand |
| Use Case | Small/medium datasets | Very large datasets |

## Troubleshooting

- If having problems in resolving the CondaPkg environment, try to set `ENV["JULIA_CONDAPKG_OPENSSL_VERSION"] = true`before loading the package. See more details [here](https://github.com/JuliaPy/CondaPkg.jl?tab=readme-ov-file#preferences)
