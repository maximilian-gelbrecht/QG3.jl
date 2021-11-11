using Documenter, QG3


makedocs(sitename="QG3", modules=[QG3], doctest=true,
pages = [
    "Home" => "index.md",
    "Reference" => Any["src/ref.md"]
    ]
    )
