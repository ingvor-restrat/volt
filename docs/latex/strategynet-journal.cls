\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{strategynet-journal}
  [2026/08/04 v2.0 StrategyNet compact research paper]

% Publication geometry and typography follow EdgeNet FMP Revision 2.0.
\PassOptionsToPackage{table}{xcolor}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions\relax
\LoadClass[a4paper,8pt]{extarticle}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage[a4paper,margin=0.62in,top=0.68in,bottom=0.68in]{geometry}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}

\RequirePackage{adjustbox}
\RequirePackage{multicol}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{float}
\RequirePackage{needspace}
\RequirePackage{listings}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning,shapes.geometric}
\RequirePackage{xparse}
\RequirePackage{xurl}
\RequirePackage{hyperref}

\definecolor{darkgreen}{rgb}{0,0.6,0}
\definecolor{darkred}{rgb}{0.6,0,0}
\definecolor{lightblue}{rgb}{0.25,0.35,0.75}
\definecolor{darkblue}{rgb}{0.05,0.15,0.42}

\DeclareCaptionFont{blue}{\color{darkblue}}
\captionsetup{
  labelfont={blue,bf},
  justification=raggedright,
  singlelinecheck=false
}
\hypersetup{
  colorlinks=true,
  linkcolor=darkblue,
  citecolor=darkgreen,
  urlcolor=darkblue,
  bookmarksopen=true,
  bookmarksnumbered=true,
  pdfdisplaydoctitle=true
}
\urlstyle{same}
\graphicspath{{assets/}{figures/}}

\setlist[itemize]{
  itemsep=0.2em,
  parsep=0pt,
  topsep=0pt,
  partopsep=0pt,
  leftmargin=2em
}
\setlist[enumerate]{
  itemsep=0.2em,
  parsep=0pt,
  topsep=0.2em,
  partopsep=0pt,
  leftmargin=2em
}
\renewcommand{\arraystretch}{1.2}
\setlength{\emergencystretch}{1.2em}
\clubpenalty=9000
\widowpenalty=9000
\displaywidowpenalty=9000
\raggedbottom

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}

% The user-facing product identity is the only running-head content.
\makeatletter
\def\@snheader{strategynet.ai}
\newcommand{\SNHeader}[1]{\gdef\@snheader{#1}}
\newcommand{\SNShortTitle}[1]{\gdef\@snheader{#1}}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\@snheader}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{12pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyhead[L]{\footnotesize\@snheader}
  \fancyfoot[C]{\thepage}
  \renewcommand{\footrulewidth}{0pt}
}
\makeatother

% As in Revision 2.0, contents use both columns on the opening page.
\newcommand{\SNtwocolumntableofcontents}{%
  \begin{multicols}{2}
    \tableofcontents
  \end{multicols}
  \thispagestyle{fancy}
  \clearpage
}
\newcommand{\SNtableofcontents}{\SNtwocolumntableofcontents}

% Column changes stay explicit so wide figures and tables can remain full width.
\newenvironment{SNcolumns}
  {\begin{multicols}{2}\raggedcolumns}
  {\end{multicols}}
\newcommand{\SNcolumnbreak}{\columnbreak}
\newcommand{\SNsectionpage}{\clearpage}

\endinput
