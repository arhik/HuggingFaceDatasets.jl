using Flux, Zygote
using Random, Statistics
using Flux.Losses: logitcrossentropy
using Flux: onecold
using HuggingFaceDatasets
using MLUtils

function text_transform(batch)
    # Basic text preprocessing for sentiment analysis
    text = lowercase.(batch["text"])
    # Tokenize by splitting on whitespace (simple tokenization)
    tokens = [split(t, r"\s+") for t in text]
    # Convert to numerical format (simple approach - use length)
    features = Float32.(length.(tokens))
    labels = Float32.(batch["label"])
    return (features, labels)
end

function streaming_loss_and_accuracy(data_loader, model, device)
    acc = 0.0f0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num += length(x)
    end
    return ls / num, acc / num
end

function train_streaming(epochs)
    batchsize = 16
    nhidden = 64
    device = cpu

    # Load streaming dataset - memory efficient for large datasets
    streaming_data = load_dataset("imdb", split="train", streaming=true).with_format("julia")

    # Create streaming pipeline: filter positive reviews -> take first 1000 -> batch
    training_data = Base.filter(sample -> sample["label"] == 1, streaming_data) |>
                    HuggingFaceDatasets.take(1000) |>
                    MLUtils.mapobs(text_transform)

    train_loader = Flux.DataLoader(training_data; batchsize, shuffle=false)

    # Simple model for text classification based on text length
    model = Chain(
        Dense(1, nhidden, relu),
        Dense(nhidden, nhidden, relu),
        Dense(nhidden, 2)  # Binary classification: positive/negative
    ) |> device

    opt = Flux.setup(AdamW(1e-3), model)

    function report(epoch)
        train_loss, train_acc = streaming_loss_and_accuracy(train_loader, model, device)
        r(x) = round(x, digits=3)
        r(x::Int) = x
        @info map(r, (; epoch, train_loss, train_acc))
    end

    report(0)
    @time for epoch in 1:epochs
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            loss, grads = withgradient(model -> logitcrossentropy(model(x), y), model)
            Flux.update!(opt, model, grads[1])
        end
        report(epoch)
    end
end

# Demonstrate streaming capabilities
function demonstrate_streaming()
    # Load streaming dataset
    imdb = load_dataset("imdb", split="train", streaming=true)
    imdb_julia = imdb.with_format("julia")

    # Show streaming operations
    first_sample = first(imdb_julia)
    println("First review label: ", first_sample["label"])

    # Take first 5 samples
    samples = collect(HuggingFaceDatasets.take(imdb_julia, 5))
    sample_labels = [s["label"] for s in samples]
    println("First 5 labels: ", sample_labels)

    # Filter positive reviews
    positive = Base.filter(sample -> sample["label"] == 1, imdb_julia)
    positive_samples = collect(HuggingFaceDatasets.take(positive, 3))
    println("Positive review count: ", length(positive_samples))

    # Create batches
    batched = HuggingFaceDatasets.batch(imdb_julia, 4)
    first_batch = first(batched)
    println("Batch size: ", length(first_batch["label"]))

    # Skip and shuffle
    skipped = HuggingFaceDatasets.skip(imdb_julia, 50)
    shuffled = HuggingFaceDatasets.shuffle(imdb_julia, seed=42, buffer_size=50)
    println("Skipped sample label: ", first(skipped)["label"])
    println("Shuffled first label: ", first(shuffled)["label"])
end

@time demonstrate_streaming()
@time train_streaming(3)

# Memory efficiency: Processes large datasets without loading everything into RAM
# Compare with regular dataset: Regular Dataset loads all data at once, Streaming loads on-demand
