using StableLinearAlgebra
using Documenter
using LinearAlgebra

DocMeta.setdocmeta!(StableLinearAlgebra, :DocTestSetup, :(using StableLinearAlgebra); recursive=true)

makedocs(;
    modules=[StableLinearAlgebra],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/cohensbw/StableLinearAlgebra.jl/blob/{commit}{path}#{line}",
    sitename="StableLinearAlgebra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cohensbw.github.io/StableLinearAlgebra.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/cohensbw/StableLinearAlgebra.jl",
    devbranch="master",
)
