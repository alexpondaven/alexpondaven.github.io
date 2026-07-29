---
layout: page
permalink: /publications/
title: Publications
description: See my Google Scholar profile for the full list of publications
years: [2026, 2025, 2022]
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
