/* Safe response rendering: generated HTML and scripts are displayed as text. */
(() => {
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const fences = () => /^ {0,3}(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)^ {0,3}\1[ \t]*(?:\n|$)/gm;

  async function copy(text, button) {
    const previous = button.textContent;
    try {
      try { await navigator.clipboard.writeText(text); }
      catch {
        const field = el("textarea");
        field.value = text;
        field.className = "response-copy-buffer";
        document.body.append(field);
        field.select();
        const ok = document.execCommand("copy");
        field.remove();
        button.focus();
        if (!ok) throw Error("Select the text and copy it manually.");
      }
      button.textContent = "Copied";
    } catch { button.textContent = "Copy unavailable"; }
    setTimeout(() => { button.textContent = previous; }, 1800);
  }

  function copyButton(text, label = "Copy code") {
    const button = el("button", "response-action", label);
    button.type = "button";
    button.onclick = () => copy(text, button);
    return button;
  }

  function downloadButton(message, file, notice) {
    const button = el("button", "response-action response-download", "Download");
    button.type = "button";
    button.setAttribute("aria-label", `Download ${file.name}`);
    button.onclick = async () => {
      button.disabled = true;
      notice.textContent = "Preparing download…";
      try {
        const response = await fetch(`/conversations/${message.conversation_id}/messages/${message.id}/artifacts/${file.index}`);
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw Error(error.error || error.detail || "Download unavailable. Reopen this chat and try again.");
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = el("a");
        link.href = url;
        link.download = file.name;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        notice.textContent = "Download ready";
      } catch (error) { notice.textContent = error.message; }
      finally { button.disabled = false; }
    };
    return button;
  }

  function inline(parent, text) {
    const pattern = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|https?:\/\/[^\s<>()]+|\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
    let end = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(document.createTextNode(text.slice(end, match.index)));
      if (match[3]) parent.append(el("strong", "", match[3]));
      else if (match[4]) parent.append(el("code", "inline-code", match[4]));
      else {
        const value = (match[2] || match[0]).replace(/[.,;!?\]\}"']+$/, "");
        let url;
        try { url = new URL(value); } catch { /* Keep malformed links as text. */ }
        if (url && !url.username && !url.password) {
          const a = el("a", "", match[1] || value);
          a.href = value; a.target = "_blank"; a.rel = "noopener noreferrer";
          parent.append(a);
          if (!match[2]) parent.append(document.createTextNode(match[0].slice(value.length)));
        } else parent.append(document.createTextNode(match[0]));
      }
      end = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(end)));
  }

  function prose(parent, text) {
    if (!text.trim()) return;
    const section = el("div", "response-prose");
    // A small Markdown subset; no HTML parsing, remote images or embedded frames.
    for (const paragraph of text.trim().split(/\n\s*\n/)) {
      const heading = paragraph.match(/^(#{1,6})\s+([^\n]+)$/);
      const node = el(heading ? `h${Math.min(heading[1].length + 1, 6)}` : "p");
      inline(node, heading ? heading[2] : paragraph);
      section.append(node);
    }
    parent.append(section);
  }

  function render(message) {
    const root = el("div", "response-body");
    const notice = el("div", "response-notice");
    notice.setAttribute("role", "status");
    notice.setAttribute("aria-live", "polite");
    const files = message.artifacts || [];
    let end = 0, index = 0;
    for (const match of message.content.matchAll(fences())) {
      prose(root, message.content.slice(end, match.index));
      const file = files.find(f => f.kind === "code" && f.code_index === index);
      const block = el("section", "response-code");
      const bar = el("div", "response-code-bar");
      bar.append(el("span", "response-filename", file?.name || match[2].trim() || "Code"));
      const actions = el("div", "response-actions");
      actions.append(copyButton(match[3]));
      if (file) actions.append(downloadButton(message, file, notice));
      bar.append(actions);
      const pre = el("pre"), code = el("code", "", match[3]);
      pre.tabIndex = 0;
      pre.setAttribute("aria-label", `${file?.name || "Code"} content`);
      pre.append(code); block.append(bar, pre); root.append(block);
      end = match.index + match[0].length;
      index++;
    }
    prose(root, message.content.slice(end));
    for (const file of files.filter(f => f.kind === "document" || f.code_index === undefined)) {
      const card = el("div", "response-file");
      const details = el("div", "response-file-details");
      details.append(el("strong", "", file.name));
      details.append(el("span", "", file.name.endsWith(".docx") ? "Word document · Ready to save" : "File · Ready to save"));
      card.append(details, downloadButton(message, file, notice)); root.append(card);
    }
    const footer = el("div", "response-footer");
    footer.append(copyButton(message.content, "Copy response"), notice);
    root.append(footer);
    return root;
  }
  window.NanaResponses = { render };
})();
