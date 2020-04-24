---
layout: page
title: "DeepDados - Blog"
subtitle: "Projects - Artificial Intelligence"
css: "/css/index.css"
meta-title: "Projects - AI (Authors: César Pedrosa Soares e Lucas Pedrosa Soares"
meta-description: "Projects - AI (Authors: César Pedrosa Soares e Lucas Pedrosa Soares"
---

<div class="list-filters">
  <span class="list-filter filter-selected"> COVID-19 Project and Artificial Intelligence </span>
</div>

<div class="posts-list">
  {% for post in site.tags.COVID %}
  <article>
    <a class="post-preview" href="{{ post.url | prepend: site.baseurl }}">
	    <h2 class="post-title">{{ post.title }}</h2>
	
	    {% if post.subtitle %}
	    <h3 class="post-subtitle">
	      {{ post.subtitle }}
	    </h3>
	    {% endif %}
      <p class="post-meta">
        Posted on {{ post.date | date: "%B %-d, %Y" }}
      </p>

      <div class="post-entry">
        {{ post.content | truncatewords: 50 | strip_html | xml_escape}}
        <span href="{{ post.url | prepend: site.baseurl }}" class="post-read-more">[Read&nbsp;More]</span>
      </div>
    </a>  
   </article>
  {% endfor %}
</div>
