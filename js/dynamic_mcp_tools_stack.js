import { app } from "../../scripts/app.js";

const TARGET_NODE_CLASS = "MCPToolsStack";
const PREFIX = "tool_";
const TOOL_TYPE = "MCP_TOOL";
const TypeSlot = { Input: 1, Output: 2 };

function isToolInput(input) {
    return input && typeof input.name === "string" && input.name.startsWith(PREFIX);
}

function renumberToolInputs(node) {
    let index = 1;
    for (const input of node.inputs || []) {
        if (!isToolInput(input)) continue;
        const name = `${PREFIX}${index}`;
        input.name = name;
        input.label = name;
        input.type = TOOL_TYPE;
        index += 1;
    }
}

function compactToolInputs(node) {
    if (!node.inputs) node.inputs = [];

    for (let i = node.inputs.length - 1; i >= 0; i -= 1) {
        const input = node.inputs[i];
        if (isToolInput(input) && input.link == null) {
            node.removeInput(i);
        }
    }

    renumberToolInputs(node);

    const toolInputs = (node.inputs || []).filter(isToolInput);
    const nextName = `${PREFIX}${toolInputs.length + 1}`;
    node.addInput(nextName, TOOL_TYPE);

    node.setDirtyCanvas?.(true, true);
}

function ensureInitialToolInput(node) {
    if (!node.inputs) node.inputs = [];
    if (!(node.inputs || []).some(isToolInput)) {
        node.addInput(`${PREFIX}1`, TOOL_TYPE);
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "workflow_knives.dynamic_mcp_tools_stack",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== TARGET_NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            ensureInitialToolInput(this);
            setTimeout(() => compactToolInputs(this), 0);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, linkInfo, data) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (slotType !== TypeSlot.Input) return result;

            const input = this.inputs?.[slot];
            if (!isToolInput(input)) return result;

            compactToolInputs(this);
            return result;
        };
    },
});
