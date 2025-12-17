// lib/services/ia/server_extractor_rest.dart
// NOTE: This now uses GraphQL (despite the filename)

import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
// import 'package:gestion_asistencia_docente/env.dart'; // Uncomment if needed

// --- KANBAN MODELS ---
class IAProjectDraft {
  String nombre;
  String? descripcion;
  String estado;
  String? fechaInicio;
  String? fechaFin;
  List<IAColumnDraft> columnas;
  IAProjectDraft({
    required this.nombre,
    this.descripcion,
    this.estado = 'Activo',
    this.fechaInicio,
    this.fechaFin,
    required this.columnas,
  });
}
class IAColumnDraft {
  String nombre;
  int orden;
  List<IATaskDraft> tareas;
  IAColumnDraft({required this.nombre, required this.orden, required this.tareas});
}
class IATaskDraft {
  String titulo;
  String? descripcion;
  String? prioridad;
  int? puntos;
  String? fechaLimite;
  IATaskDraft({required this.titulo, this.descripcion, this.prioridad, this.puntos, this.fechaLimite});
}

// --- UML MODELS ---
class UMLResponse {
  String method;
  List<UMLClass> classes;
  String generatedCode;
  String? error;

  UMLResponse({
    required this.method,
    required this.classes,
    required this.generatedCode,
    this.error,
  });

  factory UMLResponse.fromJson(Map<String, dynamic> json) {
    return UMLResponse(
      method: json['method'] ?? 'Unknown',
      classes: (json['classes'] as List? ?? [])
          .map((e) => UMLClass.fromJson(e))
          .toList(),
      generatedCode: json['generatedCode'] ?? '',
      error: json['error'],
    );
  }
}

class UMLClass {
  String name;
  List<String> attributes;
  List<String> methods;

  UMLClass({
    required this.name,
    required this.attributes,
    required this.methods,
  });

  factory UMLClass.fromJson(Map<String, dynamic> json) {
    return UMLClass(
      name: json['name'] ?? 'Unknown',
      attributes: (json['attributes'] as List? ?? []).map((e) => e.toString()).toList(),
      methods: (json['methods'] as List? ?? []).map((e) => e.toString()).toList(),
    );
  }
}


class ServerExtractorService {
  // Point to the GraphQL endpoint
  // Use 10.0.2.2 for Android Emulator, localhost for iOS/Web, or actual IP for devices
  static const _baseUrl = 'http://10.0.2.2:5000/graphql';

  // --- KANBAN EXTRACTION ---
  Future<IAProjectDraft> extractKanbanFromImage(Uint8List imageBytes) async {
    const String query = r'''
      mutation ExtractKanban($image: String!) {
        extractKanban(imageBase64: $image)
      }
    ''';

    final variables = {
      'image': base64Encode(imageBytes),
    };

    try {
      final decoded = await _postGraphQL(query, variables);
      // The result is under data -> extractKanban (which is JSON scalar)
      final kanbanJson = decoded['data']?['extractKanban'];
      if (kanbanJson == null) {
         throw Exception('GraphQL returned no data for extractKanban');
      }
      return _mapToKanbanDraft(kanbanJson);
    } catch (e) {
      print('Kanban Extraction Error: $e');
      rethrow;
    }
  }

  // --- UML EXTRACTION ---
  Future<UMLResponse> generateUMLFromImage(Uint8List imageBytes, {bool useGemini = false}) async {
    const String query = r'''
      mutation GenerateUML($image: String!, $useGemini: Boolean!) {
        generateUml(imageBase64: $image, useGemini: $useGemini) {
          method
          error
          generatedCode
          classes {
            name
            attributes
            methods
          }
        }
      }
    ''';

    final variables = {
      'image': base64Encode(imageBytes),
      'useGemini': useGemini,
    };

    try {
      final decoded = await _postGraphQL(query, variables);
      final umlData = decoded['data']?['generateUml'];
      
      if (umlData == null) {
         throw Exception('GraphQL returned no data for generateUml');
      }
      
      return UMLResponse.fromJson(umlData);
    } catch (e) {
      print('UML Generation Error: $e');
      rethrow;
    }
  }

  // --- HELPER: GRAPHQL POST ---
  Future<Map<String, dynamic>> _postGraphQL(String query, Map<String, dynamic> variables) async {
    final uri = Uri.parse(_baseUrl);
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'query': query,
        'variables': variables,
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Server Error ${response.statusCode}: ${response.body}');
    }

    final decoded = jsonDecode(response.body);
    if (decoded['errors'] != null) {
      throw Exception('GraphQL Errors: ${decoded['errors']}');
    }

    return decoded;
  }

  // --- HELPER: MAPPER FOR KANBAN ---
  static IAProjectDraft _mapToKanbanDraft(Map<String, dynamic> map) {
    String? _s(dynamic v) {
      if (v == null) return null;
      final s = v.toString().trim();
      return (s.isEmpty || s.toLowerCase() == 'null') ? null : s;
    }
    int _toInt(dynamic v) => v is int ? v : (int.tryParse(v?.toString() ?? '') ?? 0);

    final proj = map['project'] ?? {};
    final cols = (map['columns'] as List?) ?? [];

    return IAProjectDraft(
      nombre: (proj['nombre'] ?? 'Proyecto Detectado').toString(),
      descripcion: _s(proj['descripcion']),
      estado: (proj['estado'] ?? 'Activo').toString(),
      fechaInicio: _s(proj['fechaInicio']),
      fechaFin: _s(proj['fechaFin']),
      columnas: cols.map((c) {
        final tasks = (c['tasks'] as List?) ?? [];
        return IAColumnDraft(
          nombre: (c['nombre'] ?? 'Columna').toString(),
          orden: _toInt(c['orden']),
          tareas: tasks.map((t) => IATaskDraft(
            titulo: (t['titulo'] ?? 'Tarea').toString(),
            descripcion: _s(t['descripcion']),
            prioridad: _s(t['prioridad']),
            puntos: t['puntos'] == null ? null : _toInt(t['puntos']),
            fechaLimite: _s(t['fechaLimite']),
          )).cast<IATaskDraft>().toList(),
        );
      }).cast<IAColumnDraft>().toList(),
    );
  }
}
