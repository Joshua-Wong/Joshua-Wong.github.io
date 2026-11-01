// Shared by the browser and tests; the index contains published notes only.
export function searchPublishedNotes(notes, input, limit = 20) {
  const query = String(input).trim().toLocaleLowerCase();
  if (!query) return [];
  return notes.filter((note) => [note.title, note.text, note.description, ...(note.tags ?? [])]
    .some((value) => String(value ?? '').toLocaleLowerCase().includes(query)))
    .sort((a, b) => {
      const titleA = a.title.toLocaleLowerCase().includes(query);
      const titleB = b.title.toLocaleLowerCase().includes(query);
      return Number(titleB) - Number(titleA) || b.updatedAt.localeCompare(a.updatedAt) || a.slug.localeCompare(b.slug);
    })
    .slice(0, limit)
    .map((note) => {
      const index = note.text.toLocaleLowerCase().indexOf(query);
      const start = Math.max(0, index - 35);
      return {
        slug: note.slug, title: note.title, category: note.category,
        snippet: (start ? '…' : '') + note.text.slice(start, start + 160) +
          (note.text.length > start + 160 ? '…' : ''),
      };
    });
}
