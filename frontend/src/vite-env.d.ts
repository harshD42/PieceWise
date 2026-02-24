/// <reference types="vite/client" />

// Tell TypeScript that *.module.css imports are valid and return
// a plain object mapping class name strings to scoped class name strings.
declare module '*.module.css' {
  const classes: Record<string, string>
  export default classes
}