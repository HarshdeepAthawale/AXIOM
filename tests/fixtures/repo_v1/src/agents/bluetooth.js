import { preprocessInput } from '../utils/normalize.js';

export const BLUETOOTH_SETTINGS_DEEPLINK = 'app://settings/bluetooth-settings';

/**
 * Agent that opens the Bluetooth settings surface from a deeplink.
 */
export class BluetoothAgent {
  constructor(registry) {
    this.registry = registry;
    this.lastTarget = null;
  }

  /**
   * Resolve a raw deeplink string to a routable target.
   */
  parseDeeplink(raw) {
    const prepared = preprocessInput(raw);
    if (!prepared.text.startsWith('app://settings/')) {
      return null;
    }
    return prepared.text.slice('app://settings/'.length);
  }

  /**
   * Open the Bluetooth settings pane for a parsed deeplink.
   */
  openSettings(raw) {
    const target = this.parseDeeplink(raw);
    if (target !== 'bluetooth-settings') {
      return false;
    }
    this.lastTarget = target;
    return this.registry.dispatch(BLUETOOTH_SETTINGS_DEEPLINK);
  }

  /**
   * Report whether the agent can service a deeplink at all.
   */
  supports(raw) {
    const target = this.parseDeeplink(raw);
    return target !== null;
  }

  /**
   * Forget any cached routing decision.
   */
  reset() {
    this.lastTarget = null;
    this.registry.clearCache();
    return this.lastTarget;
  }
}
