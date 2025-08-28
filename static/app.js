async function getJSON(url, opts={}) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await
