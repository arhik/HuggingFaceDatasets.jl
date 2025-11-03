using Test
using HuggingFaceDatasets
using PythonCall

@testset "IterableDataset" begin
    @testset "Basic functionality" begin
        # Test loading a streaming dataset
        ds = load_dataset("mnist", split="test", streaming=true)
        @test isa(ds, IterableDataset)
        @test pyisinstance(ds.pyds, datasets.IterableDataset)

        # Test iteration
        first_item = first(ds)
        @test isa(first_item, Py)

        # Test with Julia format
        ds_julia = ds.with_format("julia")
        first_julia = first(ds_julia)
        @test isa(first_julia, Dict{String, Any})
        @test haskey(first_julia, "label")
        @test haskey(first_julia, "image")
        @test isa(first_julia["label"], Int)
        @test isa(first_julia["image"], AbstractArray)
    end

    @testset "Error handling" begin
        ds = load_dataset("mnist", split="test", streaming=true)

        # Test that random access is disabled
        @test_throws MethodError ds[0]
        @test_throws MethodError ds[1:5]
        @test_throws MethodError ds["label"]

        # Test that length is not available
        @test_throws ArgumentError length(ds)
    end

    @testset "Streaming methods" begin
        ds = load_dataset("mnist", split="test", streaming=true).with_format("julia")

        # Test take
        taken = HuggingFaceDatasets.take(ds, 3)
        @test isa(taken, IterableDataset)
        items = collect(taken)
        @test length(items) == 3

        # Test skip
        skipped = HuggingFaceDatasets.skip(ds, 5)
        first_skipped = first(skipped)
        @test isa(first_skipped, Dict{String, Any})

        # Test shuffle
        shuffled = HuggingFaceDatasets.shuffle(ds, seed=42, buffer_size=10)
        first_shuffled = first(shuffled)
        @test isa(first_shuffled, Dict{String, Any})

        # Test batch
        batched = HuggingFaceDatasets.batch(ds, 2)
        first_batch = first(batched)
        @test isa(first_batch, Dict{String, Vector})
        @test length(first_batch["label"]) == 2
        @test length(first_batch["image"]) == 2
    end

    @testset "Transform operations" begin
        ds = load_dataset("mnist", split="test", streaming=true).with_format("julia")

        # Test map
        add_label(example) = (example["new_label"] = example["label"] + 10; example)
        mapped = map(add_label, ds)
        first_mapped = first(mapped)
        @test first_mapped["new_label"] == first_mapped["label"] + 10

        # Test filter
        filtered = Base.filter(example -> example["label"] == 7, ds)
        filtered_items = collect(HuggingFaceDatasets.take(filtered, 10))
        @test all(item -> item["label"] == 7, filtered_items)

        # Test shard
        sharded = HuggingFaceDatasets.shard(ds, num_shards=2, index=0)
        first_sharded = first(sharded)
        @test isa(first_sharded, Dict{String, Any})
    end

    @testset "Julia transformations" begin
        ds = load_dataset("mnist", split="test", streaming=true)

        # Test custom julia transform
        custom_transform(example) = Dict("custom_label" => pyconvert(Int, example["label"]) * 2)
        ds_custom = with_jltransform(ds, custom_transform)
        first_custom = first(ds_custom)
        @test isa(first_custom, Dict{String, Any})
        @test first_custom["custom_label"] isa Int
    end
end
