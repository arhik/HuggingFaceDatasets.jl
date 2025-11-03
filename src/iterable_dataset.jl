"""
    IterableDataset

A Julia wrapper around an object of the python `datasets.IterableDataset` class.

Provides:
- Iteration interface for streaming datasets
- All python class' methods from `datasets.IterableDataset`.

See also [`load_dataset`](@ref) and [`DatasetDict`](@ref).

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true)
IterableDataset({
    features: ['image', 'label'],
    num_shards: 1
})

julia> for item in ds
           println(item["label"])
           break  # Only show first item
       end
5

julia> ds = ds.with_format("julia");

julia> first_item = first(ds)
Dict{String, Any} with 2 entries:
  "label" => 5
  "image" => Gray{N0f8}[Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f8}(0.0) Gray{N0f8}(0.0); Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f8}(0.0) Gray{N0f8}(0.0); … ; Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f8}(0.0) Gray{N0f8}(0.0); Gray{N0f8}(0.0) Gray{N0f8}(0.0) … Gray{N0f8}(0.0) Gray{N0f8}(0.0)]
```
"""
mutable struct IterableDataset
    pyds::Py
    jltransform

    function IterableDataset(pyds::Py, jltransform = identity)
        @assert pyisinstance(pyds, datasets.IterableDataset)
        return new(pyds, jltransform)
    end
end

function Base.getproperty(ds::IterableDataset, s::Symbol)
    if s in fieldnames(IterableDataset)
        return getfield(ds, s)
    elseif s === :with_format
        return format -> with_format(ds, format)
    elseif s === :features
        return getproperty(getfield(ds, :pyds), s) |> py2jl
    elseif s === :num_shards
        return getproperty(getfield(ds, :pyds), s) |> py2jl
    else
        res = getproperty(getfield(ds, :pyds), s)
        if pycallable(res)
            return CallableWrapper(res)
        else
            return res |> py2jl
        end
    end
end

# IterableDataset does not support random access
function Base.getindex(ds::IterableDataset, i::Integer)
    throw(MethodError("IterableDataset does not support random access (getindex). Use iteration or first() instead."))
end

function Base.getindex(ds::IterableDataset, i::AbstractVector)
    throw(MethodError("IterableDataset does not support random access (getindex). Use iteration or collect() to get multiple items."))
end

function Base.getindex(ds::IterableDataset, i::AbstractString)
    throw(MethodError("IterableDataset does not support column access via getindex. Use iteration and access fields directly."))
end

# IterableDataset may not have a defined length
function Base.length(ds::IterableDataset)
    if pyhasattr(ds.pyds, "__len__")
        return pyconvert(Int, length(ds.pyds))
    else
        # For streaming datasets, length is not known upfront
        # We can't return nothing because collect() requires a length
        # We'll throw an informative error instead
        throw(ArgumentError("IterableDataset does not have a defined length. Use iteration or collect(take(ds, n)) to process a limited number of items."))
    end
end

# Iteration interface
function Base.iterate(ds::IterableDataset)
    iterator = pyiter(ds.pyds)
    try
        py_item = pynext(iterator)
        jl_item = ds.jltransform(py_item)
        return jl_item, iterator
    catch e
        if pyisinstance(e, pyimport("builtins").StopIteration)
            return nothing
        else
            rethrow(e)
        end
    end
end

function Base.iterate(ds::IterableDataset, state)
    iterator = state
    try
        py_item = pynext(iterator)
        jl_item = ds.jltransform(py_item)
        return jl_item, iterator
    catch e
        if pyisinstance(e, pyimport("builtins").StopIteration)
            return nothing
        else
            rethrow(e)
        end
    end
end

Base.IteratorEltype(::Type{IterableDataset}) = Base.HasEltype()
Base.eltype(::Type{IterableDataset}) = Dict{String, Any}
Base.IteratorSize(::Type{IterableDataset}) = Base.SizeUnknown()

"""
    with_format(ds::IterableDataset, format)

Return a copy of `ds` with the format set to `format`.
If format is `"julia"`, the returned dataset will be transformed
with [`py2jl`](@ref) and copyless conversion from python types
will be used when possible.

See also [`set_format!`](@ref).

# Examples

```julia
julia> ds = load_dataset("mnist", split="test", streaming=true);

julia> first(ds)  # Returns Python objects
Python dict: {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x2B5B4C1F0>, 'label': 7}

julia> ds = with_format(ds, "julia");

julia> first(ds)  # Now returns Julia objects
Dict{String, Any} with 2 entries:
  "label" => 7
  "image" => UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
```
"""
function with_format(ds::IterableDataset, format::AbstractString)
    ds = deepcopy(ds)
    return set_format!(ds, format)
end

"""
    set_format!(ds::IterableDataset, format)

Set the format of `ds` to `format`. Mutating
version of [`with_format`](@ref).
"""
function set_format!(ds::IterableDataset, format)
    if format == "julia"
        if pyhasattr(ds.pyds, "reset_format")
            ds.pyds.reset_format()
        end
        ds.jltransform = py2jl
    else
        if pyhasattr(ds.pyds, "set_format")
            ds.pyds.set_format(format)
        end
        ds.jltransform = identity
    end
    return ds
end

"""
    with_jltransform(ds::IterableDataset, transform)
    with_jltransform(transform, ds::IterableDataset)

Return a copy of `ds` with the julia transform set to `transform`.
The `transform` applies when iterating, e.g. `first(ds)` or in a `for` loop.

The transform is always applied to a single data item.
The julia transform is applied after the python transform (if any).
The python transform can be set with `ds.set_transform(pytransform)`.

If `transform` is `nothing` or `identity`, the returned dataset will not be transformed.

See also [`set_jltransform!`](@ref) for the mutating version.
"""
function with_jltransform(ds::IterableDataset, transform)
    ds = deepcopy(ds)
    return set_jltransform!(ds, transform)
end

# convenience for the do syntax
with_jltransform(transform, ds::IterableDataset) = with_jltransform(ds, transform)

"""
    set_jltransform!(ds::IterableDataset, transform)
    set_jltransform!(transform, ds::IterableDataset)

Set the julia transform of `ds` to `transform`. Mutating
version of [`with_jltransform`](@ref).
"""
function set_jltransform!(ds::IterableDataset, transform)
    if transform === nothing
        ds.jltransform = identity
    else
        ds.jltransform = transform
    end
    return ds
end

set_jltransform!(transform, ds::IterableDataset) = set_jltransform!(ds, transform)

"""
    take(ds::IterableDataset, n)

Return the first `n` examples from the dataset.

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true);

julia> head = take(ds, 2);

julia> collect(head)
2-element Vector{Dict{String, Any}}:
 Dict("label" => 5, "image" => [...])
 Dict("label" => 0, "image" => [...])
```
"""
function take(ds::IterableDataset, n::Int)
    py_result = ds.pyds.take(n)
    return IterableDataset(py_result, ds.jltransform)
end

"""
    skip(ds::IterableDataset, n)

Return the dataset with the first `n` examples skipped.

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true);

julia> tail = skip(ds, 1000);  # Skip first 1000 examples
```
"""
function skip(ds::IterableDataset, n::Int)
    py_result = ds.pyds.skip(n)
    return IterableDataset(py_result, ds.jltransform)
end

"""
    shuffle(ds::IterableDataset, seed=nothing; buffer_size=1000)

Return a shuffled version of the dataset.

# Arguments

- `seed`: Random seed for reproducible shuffling
- `buffer_size`: Size of the buffer to use for shuffling (default: 1000)

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true);

julia> shuffled_ds = shuffle(ds, seed=42, buffer_size=10_000);
```
"""
function shuffle(ds::IterableDataset; seed=nothing, buffer_size=1000)
    if seed !== nothing
        py_result = ds.pyds.shuffle(seed=seed, buffer_size=buffer_size)
    else
        py_result = ds.pyds.shuffle(buffer_size=buffer_size)
    end
    return IterableDataset(py_result, ds.jltransform)
end

"""
    batch(ds::IterableDataset, batch_size::Int; drop_last_batch=false)

Create batches of size `batch_size` from the iterable dataset.

# Arguments

- `batch_size`: Number of examples per batch
- `drop_last_batch`: Whether to drop the last batch if it's smaller than batch_size

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true);

julia> batched_ds = batch(ds, 32);

julia> first_batch = first(batched_ds)  # Will be a batch of up to 32 items
```
"""
function batch(ds::IterableDataset, batch_size::Int; drop_last_batch=false)
    py_result = ds.pyds.batch(batch_size, drop_last_batch=drop_last_batch)
    return IterableDataset(py_result, ds.jltransform)
end

"""
    filter(ds::IterableDataset, function)

Filter the dataset, keeping only the examples for which `function` returns true.

The function should accept a single example (a dict) and return a boolean.

# Examples

```julia
julia> ds = load_dataset("mnist", split="train", streaming=true);

julia> filtered_ds = filter(ds, example -> example["label"] == 5);
```
"""
function Base.filter(func::Function, ds::IterableDataset)
    # Convert Julia function to Python function
    py_func = pyfunc(func)
    py_result = ds.pyds.filter(py_func)
    return IterableDataset(py_result, ds.jltransform)
end

"""
    map(ds::IterableDataset, function; remove_columns=nothing)

Apply a function to each example in the dataset.

# Arguments

- `function`: A function that takes a dict (example) and returns a dict
- `remove_columns`: Optional list of column names to remove after mapping

# Examples

```julia
julia> ds = load_dataset("c4", "en", split="train", streaming=true);

julia> add_prefix(example) = (example["text"] = "My text: " * example["text"]; example)

julia> mapped_ds = map(ds, add_prefix);
```
"""
function map(func::Function, ds::IterableDataset; remove_columns=nothing)
    # Convert Julia function to Python function
    py_func = pyfunc(func)
    kws = []
    if remove_columns !== nothing
        push!(kws, :remove_columns => remove_columns)
    end
    py_result = ds.pyds.map(py_func; kws...)
    return IterableDataset(py_result, ds.jltransform)
end

"""
    shard(ds::IterableDataset, num_shards::Int, index::Int)

Split the dataset into `num_shards` shards and return the shard at `index`.

# Examples

```julia
julia> ds = load_dataset("amazon_polarity", split="train", streaming=true);

julia> shard = shard(ds, num_shards=2, index=0);  # Get first of 2 shards
```
"""
function shard(ds::IterableDataset, num_shards::Int, index::Int)
    py_result = ds.pyds.shard(num_shards, index)
    return IterableDataset(py_result, ds.jltransform)
end

Base.show(io::IO, ds::IterableDataset) = print(io, ds.pyds)
