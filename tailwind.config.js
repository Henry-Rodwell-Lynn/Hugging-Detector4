/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        "gap-sans": ['"Gap Sans"', "sans-serif"],
      },
    },
  },
  plugins: [],
};
