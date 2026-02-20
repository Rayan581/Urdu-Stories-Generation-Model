import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "اردو کہانی جنریٹر — Urdu Story Generator",
  description:
    "Generate beautiful Urdu stories using a Trigram Language Model with BPE Tokenizer. A literary AI experience.",
  keywords: ["Urdu", "Story Generator", "NLP", "Trigram", "BPE", "Language Model"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ur" dir="rtl">
      <head>
        <meta name="theme-color" content="#1a1025" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${geistMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}