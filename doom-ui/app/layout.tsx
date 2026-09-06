import type { Metadata } from 'next';
import { Silkscreen, Geist_Mono } from 'next/font/google';
import './globals.css';

const pixel = Silkscreen({
  variable: '--font-pixel',
  weight: ['400', '700'],
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://doomfly.example'),
  title: 'DOOMFLY',
  icons: {
    icon: [
      { url: '/favicon.svg?v=side-fly', type: 'image/svg+xml', sizes: 'any' },
      { url: '/fly-icon-96.png?v=side-fly', type: 'image/png', sizes: '96x96' },
    ],
    shortcut: '/favicon.ico?v=side-fly',
    apple: [{ url: '/apple-touch-icon.png?v=side-fly', sizes: '192x192', type: 'image/png' }],
  },
  description: 'Watch experimental training in a full fly-connectome Doom simulation. Inspect live neurons and modeled memory changes; improved survival is unproven.',
  alternates: { canonical: 'https://doomfly.example/' },
  openGraph: {
    type: 'website',
    url: 'https://doomfly.example/',
    siteName: 'DOOMFLY',
    title: 'DOOMFLY',
    description: '166,700 neurons. Live experimental training in Doom. Watch modeled memory change; improved survival is unproven.',
    locale: 'en_US',
    images: [{
      url: 'https://doomfly.example/og.png',
      width: 1738,
      height: 905,
      type: 'image/png',
      alt: 'Black-and-white pixel fly beside Fly Brain / Doom. 166,700 neurons. One live experiment.',
    }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'DOOMFLY',
    description: '166,700 neurons. Live experimental training in Doom. Watch modeled memory change; improved survival is unproven.',
    images: [{
      url: 'https://doomfly.example/og.png',
      alt: 'Black-and-white pixel fly beside Fly Brain / Doom. 166,700 neurons. One live experiment.',
    }],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${pixel.variable} ${geistMono.variable}`}
      >
        {children}
      </body>
    </html>
  );
}
