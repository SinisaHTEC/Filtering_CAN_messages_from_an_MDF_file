#!/usr/bin/env python3
"""
filter_can_messages_selectable.py

Filters CAN messages from an MDF file using a DBC file.
By default, filters all messages defined in the DBC.
Optionally, you can specify a subset of message names to filter.

Default paths:
    Input MDF:     inputs/Logging_MDF.mf4
    DBC file:      inputs/PowerTrain_MDF.dbc
    Output MDF:    outputs/Filtered_CAN.mf4

Optional filter:
    CUSTOM_FILTER_CAN_MESSAGES = ["EngineData", "GearBoxInfo"]
"""

import argparse
import os
from asammdf import MDF, Signal
import cantools
import numpy as np
from collections.abc import Iterable

# ✅ Optional: specify a subset of messages to filter
# Leave empty to use all messages from DBC
CUSTOM_FILTER_CAN_MESSAGES = ["EngineData", "EngineStatus", "EngineDataIEEE", "NM_Gateway_PowerTrain"]

# ✅ Default file paths
DEFAULT_INPUT_MDF = "inputs/Logging_MDF.mf4"
DEFAULT_DBC_FILE = "inputs/PowerTrain_MDF.dbc"
DEFAULT_OUTPUT_MDF = "outputs/Filtered_CAN.mf4"


def load_dbc(dbc_path):
    try:
        return cantools.database.load_file(dbc_path)
    except Exception as e:
        print(f"❌ Failed to load DBC: {e}")
        return None

# Apply filter only on messages selected by user
def resolve_filter_ids(dbc, custom_names=None):
    use_all = not custom_names or len(custom_names) == 0
    resolved = {}

    for msg in dbc.messages:
        if use_all or msg.name in custom_names:
            resolved[msg.frame_id] = msg

    if use_all:
        print(f"✅ No custom filter provided — using all DBC messages.")
    else:
        print(f"✅ Using custom filter: {custom_names}")
        print(f"✅ Resolved filter IDs: {[hex(i) for i in resolved.keys()]}")

    return resolved

# Recursive generator designed to normalize the structure of MDF channel locations,
# which can be nested or irregular depending on how mdf.channels_db is organized.
def normalize_locations(loc):
    if isinstance(loc, tuple) and len(loc) == 2 and all(isinstance(x, int) for x in loc):
        yield loc
    elif isinstance(loc, Iterable) and not isinstance(loc, (str, bytes)):
        for item in loc:
            yield from normalize_locations(item)
# The goal of normalize_locations(loc) is to extract all valid (group_index, channel_index)
# tuples from potentially nested or mixed structures returned by mdf.channels_db

# CAN frame decoding logic
def filter_and_decode_mdf(input_mdf_path, dbc, filter_map, output_mdf_path):
    try:
        mdf = MDF(input_mdf_path)
    except Exception as e:
        print(f"❌ Failed to load MDF: {e}")
        return

    occurrences = []
    for ch_name, loc in mdf.channels_db.items():
        if ch_name == "CAN_DataFrame":
            occurrences.extend(normalize_locations(loc))

    if not occurrences:
        print("❌ No CAN_DataFrame channels found in MDF.")
        return

    signals = {}
    timestamps = {}

    for group_index, channel_index in occurrences:
        try:
            raw = mdf.get("CAN_DataFrame", group=group_index, index=channel_index)
            if len(raw) == 0 or not hasattr(raw.samples[0], "__getitem__"):
                continue
        except Exception:
            continue

        for i in range(len(raw.timestamps)):
            sample = raw.samples[i]
            ts = raw.timestamps[i]

            try:
                arb_id = sample[5]         # Extracts the arbitration ID (CAN ID) from the 6th element of the sample tuple
                if arb_id not in filter_map:    #Skips this frame if its ID isn’t in the list of messages we want to decode.
                    continue

                dlc = sample[10] if len(sample) > 10 else 8     # Extracts the Data Length Code (DLC) — how many bytes of data are valid
                # Defaults to 8 if DLC isn’t explicitly present
                raw_data = sample[11] if len(sample) > 11 else sample[-1]
                data = bytes(raw_data[:dlc])
                msg = filter_map[arb_id]

                # Use cantools to decode the binary payload data using the DBC message msg.
                # If successful, returns a dictionary of signal names and values.This ensures we only process valid, non-empty signal data
                # If decoding fails (e.g. no signal definitions, malformed data), returns None or an empty dictionary
                decoded = msg.decode(data)
                if decoded and isinstance(decoded, dict) and decoded:
                    for name, value in decoded.items():             # Store decoded signals
                        if name not in signals:
                            signals[name] = []
                            timestamps[name] = []
                        signals[name].append(value)
                        timestamps[name].append(ts)
                elif data:                              # If decoding fails, store raw bytes instead
                    raw_name = f"RawBytes_{hex(arb_id)}"
                    raw_value = int.from_bytes(data, byteorder="big")
                    if raw_name not in signals:
                        signals[raw_name] = []
                        timestamps[raw_name] = []
                    signals[raw_name].append(raw_value)
                    timestamps[raw_name].append(ts)

            except Exception:
                continue

    if not signals:
        print("❌ No matching signals decoded.")
        print("""
ℹ️ Diagnostic Summary:
The selected messages were found in the MDF file, but no named signals were decoded using the DBC.
This often happens with diagnostic or control messages that carry raw payloads without defined signal mappings.

✅ What the script did:
- Filtered frames using your selected message names (or all DBC messages if none were specified).
- Attempted to decode each frame using cantools.
- Captured raw bytes as fallback signals (e.g. RawBytes_0x51b).

📦 Result:
You can inspect the raw data in the output MDF file. If the DBC is updated later,
this script will automatically decode new signals.
""")
        return

    # ✅ Append all signals into a single Channel Group
    new_mdf = MDF()
    signal_list = []

    for name in signals:
        try:
            samples = np.array(signals[name])
            if samples.dtype == object:
                samples = samples.astype(float)
        except Exception as e:
            print(f"⚠️ Skipping signal '{name}' — unsupported type for MDF: {e}")
            continue

        sig = Signal(
            samples=samples,
            timestamps=np.array(timestamps[name]),
            name=name,
            unit=""
        )
        signal_list.append(sig)

    new_mdf.append(signal_list)

    try:
        new_mdf.save(output_mdf_path, overwrite=True)
        print(f"✅ Filtered MDF saved to: {output_mdf_path}")
    except Exception as e:
        print(f"❌ Failed to save MDF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter CAN messages from MDF using DBC.")
    parser.add_argument("--input_mdf", default=DEFAULT_INPUT_MDF, help="Path to input MDF file")
    parser.add_argument("--dbc_file", default=DEFAULT_DBC_FILE, help="Path to DBC file")
    parser.add_argument("--output_mdf", default=DEFAULT_OUTPUT_MDF, help="Path to output MDF file")
    args = parser.parse_args()

    if not os.path.exists(args.input_mdf):
        print(f"❌ Input MDF file not found: {args.input_mdf}")
        return
    if not os.path.exists(args.dbc_file):
        print(f"❌ DBC file not found: {args.dbc_file}")
        return

    dbc = load_dbc(args.dbc_file)
    if not dbc:
        return

    filter_map = resolve_filter_ids(dbc, CUSTOM_FILTER_CAN_MESSAGES)
    filter_and_decode_mdf(args.input_mdf, dbc, filter_map, args.output_mdf)

if __name__ == "__main__":
    main()
