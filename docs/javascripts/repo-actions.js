(function () {
  "use strict";

  var repositoryUrl = "https://github.com/husichao666/AI_learning_notebook";

  function createLink(label, href, primary) {
    var link = document.createElement("a");
    link.className = "repo-action-button" + (primary ? " repo-action-button--primary" : "");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = label;
    return link;
  }

  function addRepositoryActions() {
    var article = document.querySelector("article.md-content__inner");

    if (!article || article.querySelector(".repo-actions") || article.querySelector(".kb-hero")) {
      return;
    }

    var pageHeading = article.querySelector("h1");
    var pageTitle = pageHeading ? pageHeading.textContent.trim() : document.title.split(" - ")[0];
    var issueUrl = new URL(repositoryUrl + "/issues/new");
    issueUrl.searchParams.set("title", "[内容反馈] " + pageTitle);
    issueUrl.searchParams.set("body", "页面：" + window.location.href + "\n\n问题描述：\n");

    var section = document.createElement("section");
    section.className = "repo-actions";
    section.setAttribute("aria-labelledby", "repo-actions-title");

    var copy = document.createElement("div");
    copy.className = "repo-actions__copy";

    var eyebrow = document.createElement("p");
    eyebrow.className = "repo-actions__eyebrow";
    eyebrow.textContent = "参与完善";

    var heading = document.createElement("h2");
    heading.id = "repo-actions-title";
    heading.textContent = "让这篇内容继续变好";

    var description = document.createElement("p");
    description.textContent = "如果本文有所帮助，可以在 GitHub 上收藏本仓库；如果发现表述错误、公式歧义或实现差异，欢迎提交 Issue 并附上当前页面。";

    copy.appendChild(eyebrow);
    copy.appendChild(heading);
    copy.appendChild(description);

    var buttons = document.createElement("div");
    buttons.className = "repo-actions__buttons";
    buttons.appendChild(createLink("在 GitHub 上 Star", repositoryUrl, true));
    buttons.appendChild(createLink("提交内容 Issue", issueUrl.toString(), false));

    section.appendChild(copy);
    section.appendChild(buttons);
    article.appendChild(section);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(addRepositoryActions);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addRepositoryActions);
  } else {
    addRepositoryActions();
  }
})();
