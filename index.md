---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---
Introduction to GPU programming, focused on CUDA and OpenACC two paradigms for NVIDIA accelerators.

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
>
> This lesson assumes familiarity with C Programming, CUDA is a superset of the C programming language. The basic elements of the language will be explained during the first examples.
{: .prereq}

**Testing LaTeX support in the (Lesson/Workshop) The Cauchy-Schwarz Inequality**

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$

<p id="This_Is_What_I_Want"> $$ (a-b)^2 $$</p>
<p id="First_Input"> <input id="Value_A"></p>
<p id="Second_Input"> <input id="Value_B"></p>
<p id="Output"></p>
<p id="Activate"><button onclick="RUN()">Test This out</button></p>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_HTML,http://myserver.com/MathJax/config/local/local.js">
        function RUN() {
            var a = document.getElementById("Value_A").value
            var b = document.getElementById("Value_B").value
            document.getElementById("Output").innerHTML = "$$ (" + a + "-" + b + ")^2 $$";
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
</script>

{% include links.md %}
