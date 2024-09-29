using ProgressBars

const FloatType = Float64


###
### Other Utils
###
function optional_progress_bar(content, silent::Bool)
    return silent ? content : ProgressBar(content)
end

function transpose_permutedims(A::AbstractMatrix)
    return permutedims(A, (2, 1))
end

function reshape2d(A::AbstractArray)
    @assert ndims(A) > 1
    return reshape(A, :, size(A)[end])
end


###
### CUDA Utils
###
has_cuda_lib() = false
begin
    try
        using CUDA
        global has_cuda_lib() = true
    catch
    end
end

# Execute a statement only on nodes with a GPU
macro CUDA_RUN(statement)
    quote
        if has_cuda_lib()
            $(esc(statement))
        end
    end
end

# Execute a statement only on nodes without a GPU
macro NOT_CUDA_RUN(statement)
    quote
        if !has_cuda_lib()
            $(esc(statement))
        end
    end
end

macro ALLOW_SCALAR_IF_CUDA(statement)
    if @isdefined CUDA
        return quote
            if has_cuda_lib()
                if @isdefined CUDA
                    CUDA.@allowscalar $(esc(statement))
                else
                    $(esc(statement))
                end
            end
        end
    else
        return quote
            $(esc(statement))
        end
    end
end

@inline function isCUDA(a::AbstractArray)
    return has_cuda_lib() && Base.typename(typeof(a)).wrapper == CuArray
end

function free_if_CUDA!(a::AbstractArray)
    @CUDA_RUN begin
        if isCUDA(a)
            CUDA.unsafe_free!(a)
        end
    end
    return
end