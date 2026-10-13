-- pandoc filter for README.pdf (see scripts/readme_pdf.sh):
--  * drop the badge images, the top heading and the "Contents:" line
--    (the PDF has its own title block and table of contents);
--  * replace the generated Mermaid dependency graph, which LaTeX cannot
--    render, with a note;
--  * typeset inline code so long paths can break across lines in tables.

local seen_h1 = false

function Header(el)
  if el.level == 1 and not seen_h1 then
    seen_h1 = true
    return {}
  end
end

function Image(el)
  return {}
end

function Para(el)
  local first = el.content[1]
  if first and first.t == "Strong" and pandoc.utils.stringify(first) == "Contents:" then
    return {}
  end
end

function CodeBlock(el)
  if el.classes[1] == "mermaid" then
    return pandoc.Para({pandoc.Emph(pandoc.Inlines(
      "The dependency graph (data → tasks → bundles → figures) is rendered in README.md on GitHub; " ..
      "the tables below give the same information."))})
  end
end

local escapes = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$",
  ["&"] = "\\&", ["#"] = "\\#", ["^"] = "\\^{}", ["_"] = "\\_", ["%"] = "\\%",
  ["~"] = "\\textasciitilde{}",
}

function Code(el)
  local s = el.text:gsub("[\\{}$&#^_%%~]", escapes)
  -- allow a line break after path and name separators
  s = s:gsub("([/%.,%-])", "%1\\allowbreak{}"):gsub("(\\_)", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{\\small " .. s .. "}")
end
