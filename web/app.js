const fileInput = document.getElementById("file");
const button = document.getElementById("detect");
const status = document.getElementById("status");
const result = document.getElementById("result");

let resultUrl = null;

button.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    status.textContent = "Choose an image first.";
    return;
  }

  button.disabled = true;
  result.style.display = "none";
  status.textContent = "Running detection...";

  try {
    const response = await fetch("/detect", {
      method: "POST",
      headers: {"Content-Type": file.type || "application/octet-stream"},
      body: file,
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }

    const blob = await response.blob();
    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
    }
    resultUrl = URL.createObjectURL(blob);
    result.src = resultUrl;
    result.style.display = "block";

    const count = response.headers.get("X-Detection-Count");
    const latency = response.headers.get("X-Inference-Ms");
    status.textContent =
      `${count} person(s) detected | ${latency} ms model inference`;
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
  }
});
