using StableLinearAlgebra
using Documenter
using DocumenterCitations
using LinearAlgebra

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style=:numeric
)
DocMeta.setdocmeta!(StableLinearAlgebra, :DocTestSetup, :(using StableLinearAlgebra); recursive=true)

makedocs(;
    plugins=[bib],
    modules=[StableLinearAlgebra],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/StableLinearAlgebra.jl/blob/{commit}{path}#{line}",
    sitename="StableLinearAlgebra.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/StableLinearAlgebra.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => "public_api.md",
        "Developer API" => "developer_api.md"
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/StableLinearAlgebra.jl.git",
    devbranch="master",
)
