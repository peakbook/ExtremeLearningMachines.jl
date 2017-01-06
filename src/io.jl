function save(net::ELM, fname::AbstractString; compress=true)
    jldopen(fname, "w", compress=compress) do file
        write(file, "machine", net)
    end
end

function load(fname::AbstractString)
    jldopen(fname, "r") do file
        read(file, "machine")
    end
end

