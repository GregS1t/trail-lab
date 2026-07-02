#!/bin/bash
# build_slides.sh

jupyter nbconvert --to slides --execute \
  --no-input \
  rapport_course2slide.ipynb

python3 - << 'EOF'
css = """<style>
.reveal .slides section {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    max-height: 90vh !important;
    height: 90vh !important;
}
</style>"""

with open("rapport_course2slide.slides.html", "r") as f:
    html = f.read()
html = html.replace("</head>", css + "\n</head>", 1)
with open("rapport_course2slide.slides.html", "w") as f:
    f.write(html)
print("✅ Build terminé — rapport_course2slide.slides.html")
EOF
