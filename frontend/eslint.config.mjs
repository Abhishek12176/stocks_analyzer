import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat();

const eslintConfig = [...compat.config({ extends: ["next"] })];

export default eslintConfig;
