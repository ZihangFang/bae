with open('script.js', 'r') as f:
    text = f.read()

text = text.replace("style.opacity = 0;", "style.opacity = '0';")
text = text.replace("style.opacity = 1;", "style.opacity = '1';")
text = text.replace("style.opacity = 0.6;", "style.opacity = '0.6';")
text = text.replace("style.opacity = 0.8;", "style.opacity = '0.8';")
text = text.replace("style.opacity = 0.4", "style.opacity = '0.4'")
text = text.replace("this.svg.innerHTML = '';", "while (this.svg.firstChild) { this.svg.removeChild(this.svg.firstChild); }")

with open('script.js', 'w') as f:
    f.write(text)
