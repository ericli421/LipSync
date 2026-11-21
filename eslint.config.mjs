import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";
const config = [
  {
    ignores: [
      ".next/**",
      "out/**",
      "next-env.d.ts",
      "Lipsync/**",
      ".venv*/**",
      ".tools/**",
      "external/**",
      "validation-output/**",
    ],
  },
  ...nextVitals,
  ...nextTypescript,
  // These effects synchronize camera/socket lifecycle with their external APIs.
  {
    files: [
      "components/dashboard/*.tsx",
      "components/dashboard/use-frame-processor.ts",
      "components/theme-switcher.tsx",
    ],
    rules: { "react-hooks/set-state-in-effect": "off" },
  },
];

export default config;
