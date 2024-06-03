using Documenter, QG3


makedocs(sitename="QG3", modules=[QG3], doctest=true,
pages = [
    "Home" => "index.md",
    "Reference" => "ref.md"
    ]
    )

deploydocs(
    repo = "github.com/maximilian-gelbrecht/QG3.jl.git",
    devbranch = "main",
)