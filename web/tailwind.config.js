/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#fdf4f3',
          100: '#fce7e4',
          200: '#fbd3ce',
          300: '#f6b3ab',
          400: '#ef8a7c',
          500: '#e35d4a',
          600: '#d04430',
          700: '#ae3625',
          800: '#903022',
          900: '#782d22',
          950: '#41140d',
        },
        accent: {
          50: '#f0fdf6',
          100: '#dcfce9',
          200: '#bbf7d4',
          300: '#86efb3',
          400: '#4ade88',
          500: '#22c564',
          600: '#16a34f',
          700: '#158040',
          800: '#166536',
          900: '#14532e',
          950: '#052e17',
        },
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        display: ['var(--font-playfair)', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
};
