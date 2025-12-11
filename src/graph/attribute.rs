use crate::{
    core::{InfersError, InfersResult},
    onnx::{AttributeProto, attribute_proto::AttributeType},
};

/// Represents the value of an ONNX attribute.
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub enum AttributeValue {
    /// A 64-bit integer attribute.
    Int64(i64),
    /// A 32-bit floating-point attribute.
    Float(f32),
    /// A list of 64-bit integer attributes.
    VecInt64(Vec<i64>),
}

/// Represents an ONNX attribute, consisting of its name and value.
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Attribute {
    /// The name of the attribute.
    pub name: String,
    /// The value of the attribute.
    pub value: AttributeValue,
}

impl TryFrom<&AttributeProto> for Attribute {
    type Error = InfersError;

    fn try_from(attr_proto: &AttributeProto) -> InfersResult<Attribute> {
        let name = attr_proto.name();

        let value = match attr_proto.r#type() {
            AttributeType::Float => AttributeValue::Float(attr_proto.f()),
            AttributeType::Int => AttributeValue::Int64(attr_proto.i()),
            AttributeType::Ints => AttributeValue::VecInt64(attr_proto.ints.clone()),
            _ => Err(InfersError::Parse("Unsupported attribute type".to_string()))?,
        };

        Ok(Self {
            name: name.to_string(),
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_from_float() {
        let attr_proto = AttributeProto {
            name: Some("test".to_string()),
            r#type: Some(AttributeType::Float.into()),
            f: Some(1.0),
            ..Default::default()
        };

        let attr = Attribute::try_from(&attr_proto).unwrap();

        assert_eq!(attr.name, "test");
        assert_eq!(attr.value, AttributeValue::Float(1.0));
    }

    #[test]
    fn test_attribute_from_int() {
        let attr_proto = AttributeProto {
            name: Some("test".to_string()),
            r#type: Some(AttributeType::Int.into()),
            i: Some(1),
            ..Default::default()
        };

        let attr = Attribute::try_from(&attr_proto).unwrap();

        assert_eq!(attr.name, "test");
        assert_eq!(attr.value, AttributeValue::Int64(1));
    }

    #[test]
    fn test_attribute_from_ints() {
        let attr_proto = AttributeProto {
            name: Some("test".to_string()),
            r#type: Some(AttributeType::Ints.into()),
            ints: vec![1, 2, 3],
            ..Default::default()
        };

        let attr = Attribute::try_from(&attr_proto).unwrap();

        assert_eq!(attr.name, "test");
        assert_eq!(attr.value, AttributeValue::VecInt64(vec![1, 2, 3]));
    }

    #[test]
    fn test_attribute_err() {
        let attr_proto = AttributeProto {
            name: Some("test".to_string()),
            r#type: Some(AttributeType::String.into()),
            ..Default::default()
        };

        let attr_result = Attribute::try_from(&attr_proto);

        assert!(attr_result.is_err());
        assert_eq!(
            attr_result.unwrap_err().to_string(),
            "Parse error: Unsupported attribute type"
        );
    }
}
