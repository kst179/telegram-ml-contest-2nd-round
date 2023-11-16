from __future__ import annotations
from enum import Enum

class BinLabels(Enum):
    OTHER = 0
    CODE = 1

    @staticmethod
    def from_string(s: str) -> BinLabels:
        if s == "OTHER":
            return BinLabels.OTHER
        elif s == "CODE":
            return BinLabels.CODE

        raise ValueError(f"string should be one of CODE, OTHER, found {s}")
    
    @staticmethod
    def to_string(e: int | BinLabels) -> str:
        if isinstance(e, int):
            e = BinLabels(e)

        return "OTHER" if e == BinLabels.OTHER else "CODE"
    
    @staticmethod
    def from_lang(l: str) -> BinLabels:
        l = Languages.from_string(l)
        
        if l == Languages.OTHER:
            return BinLabels.OTHER
        
        return BinLabels.CODE

class Languages(Enum):
    OTHER = 0
    C = 1
    CPLUSPLUS = 2
    CSHARP = 3
    CSS = 4
    DART = 5
    DOCKER = 6
    FUNC = 7
    GO = 8
    HTML = 9
    JAVA = 10
    JAVASCRIPT = 11
    JSON = 12
    KOTLIN = 13
    LUA = 14
    NGINX = 15
    OBJECTIVE_C = 16
    PHP = 17
    POWERSHELL = 18
    PYTHON = 19
    RUBY = 20
    RUST = 21
    SHELL = 22
    SOLIDITY = 23
    SQL = 24
    SWIFT = 25
    TL = 26
    TYPESCRIPT = 27
    XML = 28

    @staticmethod
    def from_string(s):
        if s in string_to_enum:
            return string_to_enum[s]
        return Languages.OTHER

    @staticmethod
    def to_string(e):
        if isinstance(e, int):
            e = Languages(e)

        return enum_to_string[e]


enum_to_string = {
    Languages.OTHER: "Other",
    Languages.C: "C",
    Languages.CPLUSPLUS: "C++",
    Languages.CSHARP: "C#",
    Languages.CSS: "CSS",
    Languages.DART: "Dart",
    Languages.DOCKER: "Docker",
    Languages.FUNC: "FunC",
    Languages.GO: "Go",
    Languages.HTML: "HTML",
    Languages.JAVA: "Java",
    Languages.JAVASCRIPT: "JavaScript",
    Languages.JSON: "JSON",
    Languages.KOTLIN: "Kotlin",
    Languages.LUA: "Lua",
    Languages.NGINX: "NGINX",
    Languages.OBJECTIVE_C: "Objective-C",
    Languages.PHP: "PHP",
    Languages.POWERSHELL: "PowerShell",
    Languages.PYTHON: "Python",
    Languages.RUBY: "Ruby",
    Languages.RUST: "Rust",
    Languages.SHELL: "Shell",
    Languages.SOLIDITY: "Solidity",
    Languages.SQL: "SQL",
    Languages.SWIFT: "Swift",
    Languages.TL: "TL",
    Languages.TYPESCRIPT: "TypeScript",
    Languages.XML: "XML",
}

string_to_enum = {s: e for e, s in enum_to_string.items()}
