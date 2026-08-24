const ACRONYMS = new Set(['knn', 'svr', 'svm', 'arima', 'sarima', 'sarimax', 'var']);

export function formatModelName(name: string): string {
  return name
    .split('_')
    .map((word) =>
      ACRONYMS.has(word) ? word.toUpperCase() : word.charAt(0).toUpperCase() + word.slice(1)
    )
    .join(' ');
}
