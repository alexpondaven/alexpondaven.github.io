---
layout: page
permalink: /publications/
title: Publications
description: See Google Scholar for full list of publications
years: [2024, 2022] #, 1956, 1950, 1935, 1905]
nav: true
nav_order: 3
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
