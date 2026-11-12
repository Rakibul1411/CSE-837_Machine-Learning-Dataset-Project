const ACRONYMS = new Set(['knn', 'svr', 'svm', 'arima', 'sarima', 'sarimax', 'var']);

export const MONTH_NAMES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

export const MONTH_OPTIONS = MONTH_NAMES.map((name, index) => ({
  value: index + 1,
  name: name,
  shortName: name.substring(0, 3),
}));

export function getMonthName(monthNumber: number): string {
  return MONTH_NAMES[monthNumber - 1] || `Month ${monthNumber}`;
}

export function getMonthShortName(monthNumber: number): string {
  return MONTH_NAMES[monthNumber - 1]?.substring(0, 3) || `M${monthNumber}`;
}

export function formatModelName(name: string): string {
  return name
    .split('_')
    .map((word) =>
      ACRONYMS.has(word) ? word.toUpperCase() : word.charAt(0).toUpperCase() + word.slice(1)
    )
    .join(' ');
}

