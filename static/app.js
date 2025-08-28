async function registerRefs() {
  const resp = await fetch("/refs/register", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({refs: ["face1", "face2"]})
  });
  const data = await resp.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}

async function sortPhotos() {
  const resp = await fetch("/sort", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({inbox: ["img001.jpg", "img002.jpg"]})
  });
  const data = await resp.json();
  document.getElementById("output").innerText = JSON.stringify(data, null, 2);
}