using Pkg
Pkg.add("ProgressMeter")
Pkg.add("Distributions")
Pkg.add("KernelDensity")
Pkg.add("Plots")
Pkg.add("GLM")
Pkg.add("DataFrames")

function find_kde_estimation(data::AbstractVector{<:Real}, grid::AbstractVector{<:Real}, k::Function; threshold::Real=Inf)
    N = length(data)
    data = sort(data)
    values = similar(grid)
    l = 1
    r = 1
    
    for i in eachindex(grid)
        x = grid[i]
        while l < N && x - data[l] > threshold
            l += 1
        end
        while r < N && data[r + 1] - x ≤ threshold
            r += 1
        end
        values[i] = sum(k.(x .- data[l:r])) / N
    end
    return values
end

using Random
using Distributions

function run_single_experiment(N::Integer, β::Real)
    data = rand(Uniform(0, 1), N)
    sort!(data)
    grid = copy(data)
    
    threshold = sqrt(2 / β * log(10^8))
    
    values = find_kde_estimation(data, grid, x->-β*x*exp(-β/2 * x^2); threshold=threshold)
    return count(i->values[i] > 0 && values[i + 1] < 0, 1:(N-1))
end

function run_parallel_experiments(N::Integer, β::Real; num_attempts::Integer=Threads.nthreads())
    tasks = [Threads.@spawn run_single_experiment(N, β) for i in 1:num_attempts]
    results = [fetch(task) for task in tasks]
    return results
end

using ProgressMeter

N = 10000
#B = 10 .^ range(log10(2), log10(1e7), length=100)
B = [2^i for i in 1:27]
num_attempts = 100
results = Dict()

@showprogress for β in B
    results[β] = run_parallel_experiments(N, β; num_attempts=num_attempts)
end

using GLM
using DataFrames
using Plots
gr()

# Assuming results and B are already defined
means = [mean(results[β]) for β in B]
std_devs = [std(results[β]) for β in B]
confidence_intervals = 1.96 .* std_devs ./ sqrt(num_attempts)

xticks = B

# Plot
plot(
    B, means, ribbon=confidence_intervals,
    fillalpha=0.3, lw=2.5, xscale=:log10, yscale=:log10,
    label="Average # of modes", legend=:topleft,
    guidefontsize=14, # Adjust this value to change label size
    tickfontsize=12   # Adjust this value to change tick label size
)

xlabel!("β", fontsize=16)  # Adjust this value to change x-label size
ylabel!("# of modes", fontsize=16)  # Adjust this value to change y-label size
# title!("# of modes per β \\nfor n=$(N)")

# Convert data to log-log space
ids = findall(x -> x ≥ 32, B)
log_betas = log.(B[ids])
log_means = log.(means[ids])

# Fit a linear regression
model = lm(@formula(y ~ x), DataFrame(x=log_betas, y=log_means))
b = coef(model)[2]  # Slope of the line in log-log space
a = exp(coef(model)[1])  # Intercept in original space

# Print the values of 'a' and 'b'
println("The coefficient (a) is: ", a)
println("The exponent (b) is: ", b)

# Generate the fitted line
fitted_y = a .* B[ids].^b
plot!(B[ids], fitted_y, lw=2.5, color=:red, linestyle=:dash, label="Fitted line", legend=:topleft)

# Save the plot as a PDF to avoid losing any resolution
savefig("modes_n=10000.pdf")




