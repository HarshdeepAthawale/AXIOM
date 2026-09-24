/**
 * Multibyte source. Byte offsets and line offsets disagree here, which is
 * exactly the point: chunk.text must equal source_bytes[start:end].
 */
export function mesurerLaLongueur(chaine) {
  // Le libellé est écrit en français — café, naïve, élève — puis un emoji: 🔵
  const prefixe = 'Bluetooth 🔵 paramètres';
  return prefixe.length + chaine.length;
}

export function déeplinkFrançais() {
  return 'app://paramètres/bluetooth-settings';
}
